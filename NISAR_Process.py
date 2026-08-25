#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NISAR_Process.py
Description: A professional-grade geospatial processing pipeline for NASA-ISRO SAR (NISAR) 
data. This script reads complex GSLC (Ground Range Localized) HDF5/NetCDF4 data, 
calculates intensity, applies multi-looking (speckle reduction), performs 
Radiometric Terrain Correction (RTC) using a Digital Elevation Model (DEM), 
converts data to Decibels (dB), and exports the results as georeferenced 
GeoTIFFs, along with a QGIS style file (.qml) that pre-sets display Gamma
so the output looks right in QGIS without manual adjustment.
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

# Import from_bounds / transform to map a destination window back to a SOURCE pixel window
# so we can slice the HDF5 dataset instead of loading it in full for every tile
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform


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

# Tuple specifying which polarization channels to extract (e.g., HH, HV polarization)
POLARIZATIONS = ("HH", "HV")

# A list of file extensions that the script will recognize as valid input files
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

# The dimensions (height/width) of the tiles used during processing to prevent memory overflow
TILE_SIZE = 512

# List of overview/pyramid levels to build for the output GeoTIFFs (for fast zooming)
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

# The desired spatial resolution in meters for the final output (target: 10m x 10m)
TARGET_PIXEL_SIZE = 10.0

# QGIS "Gamma" display value to pre-set in the output .qml style file (Layer Properties ->
# Symbology -> Gamma). This matches the value you found looks good when set manually in QGIS
# (0.1-10 range in QGIS; values < 1 darken the display, values > 1 brighten it). This only
# affects on-screen rendering in QGIS -- the dB pixel values written to the GeoTIFF are
# untouched, so the data stays scientifically valid.
QGIS_DISPLAY_GAMMA = 0.30


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


def compute_source_window(
    dst_win_bounds: tuple[float, float, float, float],
    src_transform: Affine,
    src_height: int,
    src_width: int,
    pad: int = 2,
) -> Window:
    """
    Maps a destination tile's geographic bounds back onto the SOURCE dataset's
    pixel grid, so we can read only the small chunk of the HDF5 array that a
    given output tile actually needs (instead of loading the whole scene).

    A small pixel margin ('pad') is added on every side and the result is
    clipped to the valid extent of the source array, so resampling near tile
    edges still has the neighboring pixels it needs.

    Args:
        dst_win_bounds: (left, bottom, right, top) bounds of the destination tile.
        src_transform: Affine transform of the SOURCE dataset.
        src_height: Total height (rows) of the source dataset.
        src_width: Total width (columns) of the source dataset.
        pad: Extra pixels of margin to include on each side.

    Returns:
        Window: A pixel window into the source dataset (already clipped to
        valid bounds; may have zero width/height if the tile falls entirely
        outside the source data extent).
    """
    left, bottom, right, top = dst_win_bounds

    # Convert the destination tile's geographic bounds into source pixel coordinates
    raw_window = window_from_bounds(left, bottom, right, top, transform=src_transform)

    # Round outward (floor/ceil) and pad, so we never end up short a pixel due to rounding
    row_start = math.floor(raw_window.row_off) - pad
    col_start = math.floor(raw_window.col_off) - pad
    row_stop = math.ceil(raw_window.row_off + raw_window.height) + pad
    col_stop = math.ceil(raw_window.col_off + raw_window.width) + pad

    # Clip to the valid extent of the source array so we never index out of bounds
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_stop = min(src_height, row_stop)
    col_stop = min(src_width, col_stop)

    return Window(col_start, row_start, max(0, col_stop - col_start), max(0, row_stop - row_start))


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


def write_qgis_style_file(tif_path: Path, gamma: float, vmin: float, vmax: float) -> Path:
    """
    Writes a QGIS raster layer style file (.qml) next to the exported GeoTIFF, with the
    same base filename (e.g. scene.tif -> scene.qml).

    QGIS automatically applies a same-named .qml sitting next to a raster as its default
    style the first time that raster is added to a project, so opening the .tif in QGIS
    will already show the Gamma-adjusted rendering -- no manual slider adjustment needed.

    IMPORTANT: this only affects how QGIS DISPLAYS the file. The actual dB pixel values
    written into the .tif are completely untouched, so the data stays scientifically valid
    for analysis.

    Args:
        tif_path (Path): Path to the exported GeoTIFF (used to derive the .qml path).
        gamma (float): QGIS "Gamma" display value (valid range 0.1-10). Values below 1.0
            darken the display; values above 1.0 brighten it.
        vmin (float): Minimum data value QGIS should map to black for the gray stretch.
        vmax (float): Maximum data value QGIS should map to white for the gray stretch.

    Returns:
        Path: The path to the written .qml file.
    """
    qml_path = tif_path.with_suffix(".qml")

    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandgray" opacity="1" alphaBand="-1" grayBand="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <contrastEnhancement>
        <minValue>{vmin:.6f}</minValue>
        <maxValue>{vmax:.6f}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </contrastEnhancement>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="{gamma:.3f}"/>
    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0"/>
    <rasterresampler/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""

    qml_path.write_text(qml_content, encoding="utf-8")
    return qml_path


def export_layer(
    source_file: Path, frequency: str, polarization: str, grid: h5py.Group, run_timestamp: str, logger: logging.Logger
) -> Path:
    """
    Orchestrates the memory-efficient processing of a single SAR data layer.
    
    This function uses 'Windowed Reprojection' to avoid MemoryErrors. It calculates 
    the target 10m grid geometry first, then loops through the destination 
    image in small tiles. For each tile, it pulls only the necessary pixels 
    from the source HDF5, performs resampling, intensity calculation, 
    multi-looking, terrain correction, and decibel conversion.

    Args:
        source_file (Path): The path to the input HDF5/NetCDF file.
        frequency (str): The frequency band being processed (e.g., 'frequencyA').
        polarization (str): The polarization (e.g., 'HH').
        grid (h5py.Group): The HDF5 group containing the raw dataset.
        run_timestamp (str): Timestamp string to keep filenames consistent.
        logger (logging.Logger): The active logger instance.
        
    Returns:
        Path: The file path to the successfully created GeoTIFF.
    """
    # --- 1. DATA AND GEOMETRY SETUP ---

    # Access the specific polarization dataset (e.g., HH) inside the HDF5 group
    source_dataset = grid[polarization]
    
    # Extract the Affine transform (pixel-to-coordinate mapping) and EPSG code from the HDF5 metadata
    transform, crs = coordinate_transform(grid)
    
    # Determine the dimensions (rows and columns) of the original source dataset
    height, width = source_dataset.shape
    
    # Check if the data is complex-valued (has Real and Imaginary parts)
    is_complex = source_dataset.dtype.kind == "c"

    # Calculate the bounding box (left, bottom, right, top) of the original data in geographic coordinates
    left, bottom, right, top = array_bounds(height, width, transform)
    
    # Calculate the new target grid (10m resolution) based on the bounding box and target pixel size
    # dst_transform: The new pixel-to-coordinate matrix for the 10m grid
    # dst_width/height: The total number of pixels in the final 10m output image
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs, crs, width, height, left, bottom, right, top, resolution=(TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE)
    )

    # --- 2. OUTPUT FILE PREPARATION ---

    # Construct the final filename using the source name and processing parameters
    out_filename = f"{source_file.stem}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
    
    # Define the path where the final GeoTIFF will be stored
    out_path = PROCESSED_DIRECTORY / out_filename
    
    # Create a temporary filename to prevent file corruption if the script crashes mid-process
    temp_path = out_path.with_suffix(".part.tif")
    
    # Delete the temporary file if it already exists from a previous failed attempt
    if temp_path.exists():
        temp_path.unlink()

    # Define the GeoTIFF writing profile (metadata and compression settings)
    profile = {
        "driver": "GTiff",                # Use the GeoTIFF file format
        "width": dst_width,               # Width of the final output image
        "height": dst_height,             # Height of the final output image
        "count": 1,                       # Number of bands (we are only producing 1 band: dB)
        "dtype": "float32",               # Use 32-bit floating point numbers for precision
        "crs": crs,                       # The coordinate reference system of the output
        "transform": dst_transform,        # The 10m resolution transform matrix
        "nodata": -9999.0,                # Set the value used to represent "no data" or nulls
        "tiled": True,                    # Enable internal tiling for faster GIS loading
        "blockxsize": TILE_SIZE,          # Size of internal tiles (512 pixels)
        "blockysize": TILE_SIZE,          # Size of internal tiles (512 pixels)
        "compress": "deflate",            # Use DEFLATE compression to save disk space
        "predictor": 3,                   # Use horizontal differencing (improves compression for floats)
        "BIGTIFF": "IF_SAFER",            # Support files larger than 4GB
    }

    # Attempt to open the Digital Elevation Model (DEM) if the file path exists
    dem_dataset = None
    if LOCAL_DEM_PATH.exists():
        dem_dataset = rasterio.open(LOCAL_DEM_PATH)
    else:
        logger.warning(f"DEM not found at {LOCAL_DEM_PATH}. RTC step will be skipped.")

    # --- 3. THE WINDOWED PROCESSING LOOP (The Memory-Saving Part) ---

    # Open the temporary destination file for writing
    with rasterio.open(temp_path, "w", **profile) as dst:
        
        # Set a description for the first band in the GeoTIFF metadata
        dst.set_band_description(1, f"{frequency} {polarization} Intensity Multilook RTC dB")

        # We iterate through the DESTINATION (output) grid in chunks (Tiles)
        # This ensures we only ever have ONE tile in memory at a time
        for row_off in range(0, dst_height, TILE_SIZE):
            for col_off in range(0, dst_width, TILE_SIZE):
                
                # Determine the window size for the current tile (handling the edges of the image)
                win_h = min(TILE_SIZE, dst_height - row_off)
                win_w = min(TILE_SIZE, dst_width - col_off)
                
                # Create a 'Window' object defining the current rectangular sub-region of the output
                win = Window(col_off, row_off, win_w, win_h)

                # Geographic bounds of this destination tile - used both to find the matching
                # chunk of the SOURCE dataset below and later for the DEM/RTC step.
                win_bounds = window_bounds(win, dst_transform)

                # Create a buffer to hold the resampled/reprojected data for this specific tile
                # We use float32 as a general-purpose buffer
                tile_buffer = np.empty((win.height, win.width), dtype=np.float32)

                # --- STEP A: WINDOWED REPROJECTION (Resampling) ---
                # This is the most important step. It pulls only the data needed for this 
                # specific 10m window from the source HDF5 and resamples it into the buffer.
                #
                # NOTE: source_dataset is a plain h5py.Dataset, not a rasterio/GDAL dataset,
                # so rasterio.warp.reproject() cannot do a lazy/windowed read on it the way it
                # does for the DEM (rasterio.band(dem_dataset, 1)). Passing source_dataset
                # directly forces the ENTIRE scene into memory on every tile. Instead, we work
                # out which small slice of the source array this tile needs and read only that.
                src_win = compute_source_window(win_bounds, transform, height, width)

                if src_win.width <= 0 or src_win.height <= 0:
                    # This destination tile falls completely outside the source data extent.
                    # Fill with nodata and skip straight to writing this tile.
                    db_tile = np.full((win.height, win.width), -9999.0, dtype=np.float32)
                    dst.write(db_tile, 1, window=win)
                    continue

                # Read only the required chunk from the HDF5 dataset (cheap: a few MB, not GB)
                source_chunk = source_dataset[src_win.toslices()]
                # Affine transform describing just this small chunk, for src_transform below
                src_chunk_transform = window_transform(src_win, transform)

                if is_complex:
                    # If the source is complex, we must use a complex buffer during reproject
                    # to avoid losing the imaginary component before calculating intensity.
                    complex_buffer = np.empty((win.height, win.width), dtype=np.complex64)
                    reproject(
                        source=source_chunk,
                        destination=complex_buffer,
                        src_transform=src_chunk_transform,
                        src_crs=crs,
                        dst_transform=dst.window_transform(win), # Map to the output 10m window
                        dst_crs=crs,
                        resampling=Resampling.nearest,         # Use nearest neighbor for raw data
                    )
                    # Convert the complex buffer into intensity (Real^2 + Imag^2)
                    tile_buffer = calculate_intensity(complex_buffer)
                else:
                    # If the data is already real, reproject directly into the float32 buffer
                    reproject(
                        source=source_chunk,
                        destination=tile_buffer,
                        src_transform=src_chunk_transform,
                        src_crs=crs,
                        dst_transform=dst.window_transform(win), # Map to the output 10m window
                        dst_crs=crs,
                        resampling=Resampling.nearest,
                    )

                # --- STEP B: MULTI-LOOKING (Speckle Reduction) ---
                # Apply the spatial averaging filter to reduce SAR salt-and-pepper noise
                ml_tile = apply_multilook(tile_buffer, looks=5)

                # --- STEP C: RADIOMETRIC TERRAIN CORRECTION (RTC) ---
                if dem_dataset:
                    try:
                        # (win_bounds was already computed above, before the source read)

                        # Create a temporary transform for the DEM window to match the output tile
                        win_transform = rasterio.transform.from_bounds(*win_bounds, win.width, win.height)
                        
                        # Create an empty buffer for the DEM data required for this tile
                        dem_tile = np.empty((win.height, win.width), dtype=np.float32)

                        # Pull only the necessary chunk of the DEM for this tile
                        reproject(
                            source=rasterio.band(dem_dataset, 1),
                            destination=dem_tile,
                            dst_transform=win_transform,
                            dst_crs=crs,
                            resampling=Resampling.bilinear
                        )
                        
                        # Adjust intensity based on the local terrain slope to correct for shadows
                        rtc_tile = apply_rtc(ml_tile, dem_tile, abs(dst_transform.a), abs(dst_transform.e))
                    except Exception as e:
                        # If math or coordinate errors occur during RTC, use the unfiltered tile
                        logger.debug(f"RTC failed on tile at {row_off},{col_off}: {e}. Using uncorrected intensity.")
                        rtc_tile = ml_tile
                else:
                    # If no DEM is available, skip RTC and use the multi-looked tile
                    rtc_tile = ml_tile

                # --- STEP D: DECIBEL CONVERSION ---
                # Convert the linear intensity values to the logarithmic Decibel (dB) scale
                db_tile = convert_to_decibels(rtc_tile)
                
                # Replace any non-finite numbers (like Infinity or NaN) with the designated 'nodata' value
                db_tile = np.where(np.isfinite(db_tile), db_tile, -9999.0)
                
                # --- STEP E: WRITE TO DISK ---
                # Write the finished, processed 10m tile into the destination GeoTIFF file
                dst.write(db_tile.astype(np.float32), 1, window=win)

        # Compute real min/max statistics of the written dB data (nodata pixels are excluded
        # automatically). These drive the min/max contrast stretch in the QGIS style file below.
        band_stats = dst.statistics(1, approx=False)

        # Build overview/pyramid levels (low-res versions) for fast zooming in GIS software
        dst.build_overviews(OVERVIEW_FACTORS, Resampling.average)
        
        # Add metadata tags to the file so GIS software knows how the overviews were built
        dst.update_tags(ns="rio_overview", resampling="average")

    # Close the DEM file connection to free up system resources
    if dem_dataset:
        dem_dataset.close()

    # Rename the temporary '.part.tif' file to the final intended filename
    os.replace(temp_path, out_path)
    
    # Log the completion of this specific layer
    logger.info("Created Processed dB GeoTIFF: %s", out_path)

    # Write a matching QGIS style file (.qml) so the GeoTIFF opens in QGIS already
    # rendered with the QGIS_DISPLAY_GAMMA setting -- no manual Gamma slider adjustment
    # needed. This is display-only and does not modify the dB pixel values above.
    qml_path = write_qgis_style_file(out_path, QGIS_DISPLAY_GAMMA, band_stats.min, band_stats.max)
    logger.info("Created QGIS style file: %s", qml_path)
    
    # Return the path to the finished file
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