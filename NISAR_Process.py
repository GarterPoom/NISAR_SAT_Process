#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NISAR_Process.py
Description: A professional-grade geospatial processing pipeline for NASA-ISRO SAR (NISAR) 
data. This script reads complex GSLC (Ground Range Localized) HDF5/NetCDF4 data, 
calculates intensity, applies multi-looking (speckle reduction), performs 
Radiometric Terrain Correction (RTC) using a Digital Elevation Model (DEM), 
converts data to Decibels (dB), and exports the results as georeferenced GeoTIFFs.
"""

# --- IMPORT SECTION ---

# Import future behavior to ensure type hinting works correctly in older Python 3 versions
from __future__ import annotations

# Import math for floating-point operations (used for checking pixel size equality)
import math

# Import logging to track script execution and errors in real-time and to files
import logging

# Import os for operating system tasks like replacing files (os.replace)
import os

# Import sys to interact with the interpreter (used for sys.stdout and exiting)
import sys

# Import datetime to create unique, timestamped filenames for logs and outputs
from datetime import datetime

# Import Path from pathlib for robust, cross-platform filesystem path manipulation
from pathlib import Path

# Import Iterator to type-hint that certain functions will yield values (generators)
from typing import Iterator

# Import h5py for reading scientific data in HDF5 and NetCDF4 formats
import h5py

# Import numpy for high-performance numerical array manipulations (the backbone of data math)
import numpy as np

# Import uniform_filter to perform spatial averaging (the "multi-looking" process)
from scipy.ndimage import uniform_filter

# Import rasterio for handling geospatial raster data (reading DEMs and writing GeoTIFFs)
import rasterio

# Import Resampling from rasterio to define how pixels are interpolated during resampling
from rasterio.enums import Resampling

# Import reproject and calculate_default_transform to handle coordinate/grid remapping
from rasterio.warp import calculate_default_transform, reproject

# Import array_bounds to calculate the spatial extent (left, bottom, right, top) of a raster
from rasterio.transform import array_bounds

# Import Affine to create the transformation matrix used for georeferencing pixels
from affine import Affine

# Import Window to enable "tiled" processing (reading/writing small chunks to save RAM)
from rasterio.windows import Window

# Import window_bounds to convert pixel-based window coordinates to geographic bounds
from rasterio.windows import bounds as window_bounds


# --- CONFIGURATION & CONSTANTS ---

# Determine the absolute path to the directory where this script is stored
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Root directory where the raw NISAR HDF5/NetCDF files are located
ROOT_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_Product"

# Directory where the processed, georeferenced GeoTIFF files will be saved
PROCESSED_DIRECTORY = SCRIPT_DIRECTORY / "GeoTIFF_Processed"

# Directory for storing execution logs for debugging and auditing
LOG_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_logs"

# Path to the Digital Elevation Model (DEM) used for Terrain Correction (RTC)
# This must match the CRS and spatial resolution of your target area
LOCAL_DEM_PATH = SCRIPT_DIRECTORY / "NASA_DEM" / "NISAR_DEM_1-20260817_064201_Mosaic.tif"

# The internal HDF5 path structure where the GSLC spatial grids are stored
GSLC_GRIDS_PATH = "science/LSAR/GSLC/grids"

# Tuple specifying which frequency band to process (e.g., L-Band frequencyA)
FREQUENCIES = ("frequencyA",)

# Tuple specifying which polarization channels to extract (e.g., HH polarization)
POLARIZATIONS = ("HH",)

# A list of file extensions that the script will recognize as valid input files
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

# The dimensions (height/width) of the tiles used during processing to prevent memory overflow
TILE_SIZE = 512

# List of overview/pyramid levels to build for the output GeoTIFFs (for fast zooming)
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

# The desired spatial resolution in meters for the final output (target: 10m x 10m)
TARGET_PIXEL_SIZE = 10.0


# --- FUNCTION DEFINITIONS ---

def setup_logger(log_directory: Path) -> logging.Logger:
    """
    Configures and initializes the logging system.
    
    Args:
        log_directory (Path): The path to the folder where log files will be stored.
        
    Returns:
        logging.Logger: A configured logger object that outputs to both console and file.
    """
    # Create the directory if it doesn't already exist
    log_directory.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename using the current date and time
    log_path = log_directory / f"NISAR_L_Band_Process_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    # Define the logger name
    logger = logging.getLogger("NISAR_L_Band_Process")
    logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG up to CRITICAL
    logger.propagate = False       # Prevent logs from being passed to the root logger
    logger.handlers.clear()        # Clear existing handlers to avoid duplicate logs
    
    # Standardized timestamp format for logs
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Add a StreamHandler (for terminal output) and a FileHandler (for file output)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.info("Log file: %s", log_path)
    logger.info("Detailed terminal logging is enabled.")
    return logger


def coordinate_transform(grid: h5py.Group) -> tuple[Affine, str]:
    """
    Converts HDF5 coordinate arrays into an Affine transform for GeoTIFF georeferencing.
    
    Args:
        grid (h5py.Group): The HDF5 group containing 'xCoordinates', 'yCoordinates', 
                            and 'projection' metadata.
                            
    Returns:
        tuple[Affine, str]: A tuple containing the Affine transform matrix and the EPSG CRS string.
        
    Raises:
        ValueError: If coordinate arrays are too small or if resolution is zero.
    """
    # Extract coordinate arrays from the HDF5 file
    x_coordinates = grid["xCoordinates"]
    y_coordinates = grid["yCoordinates"]
    
    # Validation: We need at least two points to calculate resolution (distance between points)
    if len(x_coordinates) < 2 or len(y_coordinates) < 2:
        raise ValueError("xCoordinates and yCoordinates must each contain at least two values")
        
    # Calculate pixel resolution by finding the difference between adjacent coordinates
    x_resolution = float(x_coordinates[1] - x_coordinates[0])
    y_resolution = float(y_coordinates[1] - y_coordinates[0])
    
    # Prevent division by zero errors
    if x_resolution == 0 or y_resolution == 0:
        raise ValueError("Coordinate spacing cannot be zero")
        
    # Retrieve the projection information from the HDF5 attributes
    projection = grid["projection"]
    # Extract the EPSG code (e.g., 4326) from the attributes or the group itself
    epsg_code = int(projection.attrs.get("epsg_code", projection[()]))
    
    # Construct the Affine transform matrix
    # Parameters: (scale_x, shear_x, origin_x, shear_y, scale_y, origin_y)
    # Note: origin is shifted by half a pixel to align the center of the pixel with the coordinate
    transform = Affine(x_resolution, 0.0, float(x_coordinates[0]) - x_resolution / 2,
                       0.0, y_resolution, float(y_coordinates[0]) - y_resolution / 2)
    
    return transform, f"EPSG:{epsg_code}"


def source_windows(height: int, width: int) -> Iterator[Window]:
    """
    A generator function that yields 'Window' objects for tiling.
    This allows processing massive rasters by only loading small chunks into RAM.
    
    Args:
        height (int): Total height of the image in pixels.
        width (int): Total width of the image in pixels.
        
    Yields:
        Window: A rasterio Window object defining a specific rectangular sub-region.
    """
    # Iterate through the image in steps of TILE_SIZE (e.g., 512 pixels)
    for row_offset in range(0, height, TILE_SIZE):
        # Ensure we don't try to grab a tile larger than the image height at the bottom edge
        tile_height = min(TILE_SIZE, height - row_offset)
        
        for column_offset in range(0, width, TILE_SIZE):
            # Ensure we don't try to grab a tile larger than the image width at the right edge
            tile_width = min(TILE_SIZE, width - column_offset)
            
            # Yield the sub-region window to the caller
            yield Window(column_offset, row_offset, tile_width, tile_height)


def calculate_intensity(complex_data: np.ndarray) -> np.ndarray:
    """
    Converts complex-valued SAR data into intensity.
    Formula: Intensity = Real^2 + Imaginary^2.
    
    Args:
        complex_data (np.ndarray): An array of complex numbers (e.g., from SAR signal).
        
    Returns:
        np.ndarray: An array of floating-point intensity values.
    """
    # Compute the square of the real component
    real_sq = np.square(np.real(complex_data)).astype(np.float32)
    # Compute the square of the imaginary component
    imag_sq = np.square(np.imag(complex_data)).astype(np.float32)
    
    # Combine components to get total intensity
    return real_sq + imag_sq


def apply_multilook(intensity: np.ndarray, looks: int = 5) -> np.ndarray:
    """
    Reduces 'speckle' noise using a spatial multi-looking (averaging) filter.
    
    Args:
        intensity (np.ndarray): The input intensity array.
        looks (int): The size of the averaging window (e.g., 5x5).
        
    Returns:
        np.ndarray: The spatially filtered intensity array.
    """
    # Use a uniform filter to perform a simple spatial average across the 'looks' window
    return uniform_filter(intensity, size=looks, mode="reflect")


def apply_rtc(intensity: np.ndarray, dem_array: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """
    Performs basic Radiometric Terrain Correction (RTC).
    Adjusts intensity based on the local terrain slope to compensate for shadows/illumination.
    
    Args:
        intensity (np.ndarray): Input intensity tile.
        dem_array (np.ndarray): Corresponding elevation tile from the DEM.
        res_x (float): Horizontal resolution.
        res_y (float): Vertical resolution.
        
    Returns:
        np.ndarray: Terrain-corrected intensity.
    """
    # Check if DEM array is valid; if not, return the original intensity
    if dem_array.shape[0] < 2 or dem_array.shape[1] < 2:
        return intensity

    # Calculate the gradient (rate of change) in Y and X directions
    dy, dx = np.gradient(dem_array, res_y, res_x)
    
    # Calculate the local slope magnitude: sqrt(slope_x^2 + slope_y^2)
    slope = np.sqrt(dx**2 + dy**2)
    
    # Calculate the cosine of the incidence angle (simplified using slope)
    # We add a tiny epsilon (1e-6) to prevent division by zero
    cos_i = np.cos(np.arctan(slope)) + 1e-6
    
    # Correct the intensity: intensity / cos(incidence_angle)
    return intensity / cos_i


def convert_to_decibels(intensity: np.ndarray) -> np.ndarray:
    """
    Converts linear intensity values to a logarithmic Decibel (dB) scale.
    Formula: 10 * log10(Intensity).
    
    Args:
        intensity (np.ndarray): Input intensity array.
        
    Returns:
        np.ndarray: Decibel-scale array.
    """
    # Replace any non-positive values with a tiny epsilon (1e-10) to avoid log10(0) errors
    safe_intensity = np.where(intensity > 0, intensity, 1e-10)
    
    # Standard log conversion to dB
    return 10.0 * np.log10(safe_intensity)


def export_layer(
    source_file: Path, frequency: str, polarization: str, grid: h5py.Group, run_timestamp: str, logger: logging.Logger
) -> Path:
    """
    The main orchestration function for a single layer. Handles reading, 
    resampling, processing (Multilook/RTC), and saving to GeoTIFF.
    
    Args:
        source_file (Path): Path to the input HDF5/NetCDF file.
        frequency (str): The frequency band being processed.
        polarization (str): The polarization being processed.
        grid (h5py.Group): The specific HDF5 group containing the grid data.
        run_timestamp (str): Timestamp used for consistent naming in a single run.
        logger (logging.Logger): The active logger instance.
        
    Returns:
        Path: The path to the newly created processed GeoTIFF.
    """
    # 1. ACCESS DATA
    source_dataset = grid[polarization]
    transform, crs = coordinate_transform(grid)
    height, width = source_dataset.shape
    is_complex = source_dataset.dtype.kind == "c"  # Check if data is complex-valued

    # Calculate pixel resolution from the transform to check if resampling is needed
    pixel_x, pixel_y = abs(transform.a), abs(transform.e)
    # If pixels are not square or don't match TARGET_PIXEL_SIZE, we must resample
    needs_resample = not math.isclose(pixel_x, pixel_y, rel_tol=1e-9, abs_tol=1e-6)

    # 2. PRE-PROCESSING (RESAMPLING)
    if needs_resample:
        logger.info("Resampling to %.0f x %.0f m.", TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE)
        full_array = source_dataset[:, :]
        
        # If complex, convert to intensity before resampling to reduce computation/complexity
        if is_complex:
            full_array = calculate_intensity(full_array)
            is_complex = False
            
        # Calculate new bounding box and transform for the target resolution
        left, bottom, right, top = array_bounds(height, width, transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, crs, width, height, left, bottom, right, top, resolution=(TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE)
        )
        
        # Prepare destination array for resampling
        resampled_array = np.empty((dst_height, dst_width), dtype=np.float32)
        # Perform the reprojection/resampling using nearest neighbor to maintain signal integrity
        reproject(
            source=full_array, destination=resampled_array,
            src_transform=transform, src_crs=crs,
            dst_transform=dst_transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
        # Update variables to use the resampled state for the rest of the function
        source_dataset = resampled_array
        transform = dst_transform
        height, width = dst_height, dst_width

    # 3. FILE SETUP
    # Construct filename: [original]_frequency_polarization_Processed_dB_timestamp.tif
    out_filename = f"{source_file.stem}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
    out_path = PROCESSED_DIRECTORY / out_filename
    # Use a temporary file name to ensure that if the script crashes, we don't leave a broken .tif
    temp_path = out_path.with_suffix(".part.tif")
    
    if temp_path.exists():
        temp_path.unlink()

    # Define GeoTIFF profile (metadata, compression, tiling)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": -9999.0,
        "tiled": True, "blockxsize": TILE_SIZE, "blockysize": TILE_SIZE,
        "compress": "deflate", "predictor": 3, "BIGTIFF": "IF_SAFER",
    }

    # Open DEM if it exists for the RTC step
    dem_dataset = None
    if LOCAL_DEM_PATH.exists():
        dem_dataset = rasterio.open(LOCAL_DEM_PATH)
    else:
        logger.warning(f"DEM not found at {LOCAL_DEM_PATH}. RTC step will be skipped.")

    # Calculate total number of tiles for progress tracking
    total_tiles = ((height + TILE_SIZE - 1) // TILE_SIZE) * ((width + TILE_SIZE - 1) // TILE_SIZE)

    # 4. TILE-BASED PROCESSING LOOP
    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.set_band_description(1, f"{frequency} {polarization} Intensity Multilook RTC dB")

        for tile_num, window in enumerate(source_windows(height, width), start=1):
            # Read a small chunk (window) of the data from the source
            tile_data = source_dataset[window.row_off:window.row_off+window.height, 
                                       window.col_off:window.col_off+window.width]
            
            # Step A: Intensity calculation (if data is still complex)
            if is_complex:
                intensity_tile = calculate_intensity(tile_data)
            else:
                intensity_tile = tile_data

            # Step B: Multi-looking (Speckle Filtering)
            ml_tile = apply_multilook(intensity_tile, looks=5)

            # Step C: Radiometric Terrain Correction (RTC)
            if dem_dataset:
                try:
                    # Calculate geographic bounds for the current tile
                    win_bounds = window_bounds(window, transform)
                    # Create a local transform for the DEM window
                    win_transform = rasterio.transform.from_bounds(*win_bounds, window.width, window.height)
                    dem_tile = np.empty((window.height, window.width), dtype=np.float32)

                    # Reproject/Sample only the piece of the DEM required for this tile
                    reproject(
                        source=rasterio.band(dem_dataset, 1),
                        destination=dem_tile,
                        dst_transform=win_transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear
                    )
                    # Apply the terrain correction formula
                    rtc_tile = apply_rtc(ml_tile, dem_tile, abs(transform.a), abs(transform.e))
                except Exception as e:
                    # If RTC fails (e.g. coordinate mismatch), fall back to uncorrected intensity
                    logger.debug(f"RTC failed on tile {tile_num}: {e}. Using uncorrected intensity.")
                    rtc_tile = ml_tile
            else:
                rtc_tile = ml_tile

            # Step D: Convert to Decibels
            db_tile = convert_to_decibels(rtc_tile)
            # Handle non-finite numbers (inf/nan) by setting them to the designated nodata value
            db_tile = np.where(np.isfinite(db_tile), db_tile, -9999.0)
            
            # Write the processed tile to the destination file
            dst.write(db_tile.astype(np.float32), 1, window=window)

            # Progress reporting
            if tile_num == total_tiles or tile_num % max(1, total_tiles // 10) == 0:
                logger.info("%s/%s progress: %.0f%% (%d/%d tiles)", frequency, polarization, 100 * tile_num / total_tiles, tile_num, total_tiles)

        # Build pyramid overviews (the low-resolution versions used for fast zooming)
        dst.build_overviews(OVERVIEW_FACTORS, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    # Close the DEM file if it was opened
    if dem_dataset:
        dem_dataset.close()

    # Move the temp file to the final destination name
    os.replace(temp_path, out_path)
    logger.info("Created Processed dB GeoTIFF: %s", out_path)
    return out_path


def main() -> None:
    """
    The main entry point. Discovers all supported products in the input directory 
    and triggers the processing pipeline for each file found.
    """
    # Initialize logging
    logger = setup_logger(LOG_DIRECTORY)
    
    # Ensure the input directory exists before proceeding
    if not ROOT_DIRECTORY.is_dir():
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        raise SystemExit(1)
        
    # Ensure output directory exists
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    # Find all files matching the supported extensions in the input tree
    source_files = [p for p in ROOT_DIRECTORY.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not source_files:
        logger.warning("No supported products found.")
        return
        
    # Create a single timestamp to use across all files in this specific execution run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Found %d product(s). Starting processing...", len(source_files))

    # Process each file one by one
    for source_file in source_files:
        try:
            # Open the HDF5 file in read mode
            with h5py.File(source_file, "r") as product:
                # Navigate to the GSLC grid section
                grids = product[GSLC_GRIDS_PATH]
                # Iterate through each requested frequency (e.g., frequencyA)
                for frequency in FREQUENCIES:
                    grid = grids[frequency]
                    # Iterate through each requested polarization (eg., HH)
                    for polarization in POLARIZATIONS:
                        # Execute the full processing and export pipeline
                        export_layer(source_file, frequency, polarization, grid, run_timestamp, logger)
        
        # Catch errors related to reading files that were interrupted during download
        except OSError as e:
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            logger.debug("Truncated file details: %s", e)
            
        # Catch and log any other unexpected errors to prevent the script from crashing mid-batch
        except Exception as e:
            logger.error("Failed to process product %s: %s", source_file.name, e)
            logger.debug("Detailed traceback:", exc_info=True)

    logger.info("Export complete.")


# --- EXECUTION START ---
if __name__ == "__main__":
    main()
