#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NISAR_Process.py

Purpose
-------
Convert supported NISAR HDF5/NetCDF4 products into tiled, georeferenced GeoTIFF
layers expressed in decibels (dB). Products with square pixels retain their native
grid; products with rectangular pixels are resampled to 10 m x 10 m before export.

Supported products
------------------
GSLC (Geocoded Single Look Complex): complex-valued polarization layers such as
HH and HV. Each tile is converted to intensity, multilooked, and optionally
corrected with a local DEM before dB conversion.

GCOV (Geocoded Polarimetric Covariance): real diagonal covariance layers such as
HHHH and HVHV. These source layers have already been multilooked and radiometrically
terrain corrected, so they are converted directly to dB without applying those
steps again.

"""

# --- IMPORT SECTION ---

# Import future behavior to ensure type hinting works correctly in older Python 3 versions
from __future__ import annotations

# Import logging to track script execution and errors in real-time and to files
import logging

# Import math for rounding rectangular-grid output dimensions up to whole 10 m pixels.
import math

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

# Import Resampling for DEM alignment and GeoTIFF overview construction
from rasterio.enums import Resampling

# Import reproject to align DEM tiles with the NISAR product grid for RTC
from rasterio.warp import reproject

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
LOCAL_DEM_PATH = SCRIPT_DIRECTORY / "Directory_DEM" / "your_local_DEM.tif"

# Internal HDF5 paths for the supported NISAR products.
PRODUCT_GRIDS_PATHS = {
    "GSLC": "science/LSAR/GSLC/grids",
    "GCOV": "science/LSAR/GCOV/grids",
}

# Tuple specifying which frequency band to process (e.g., L-Band frequencyA)
FREQUENCIES = ("frequencyA",)

# Tuple specifying which polarization channels to extract (e.g., HH, HV polarization)
POLARIZATIONS = ("HH", "HV")

# A list of file extensions that the script will recognize as valid input files
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

# The dimensions (height/width) of the tiles used during processing to prevent memory overflow
TILE_SIZE = 512

# GeoTIFF value used for pixels outside the valid NISAR swath.  This is deliberately
# outside the expected backscatter dB range, so invalid edge pixels are transparent
# in GIS software instead of appearing as a black -100 dB border.
OUTPUT_NODATA = -9999.0

# List of overview/pyramid levels to build for the output GeoTIFFs (for fast zooming)
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

# Resolution used when a source grid has rectangular (non-square) pixels.
RECTANGULAR_PIXEL_OUTPUT_RESOLUTION = 10.0


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


def grid_has_square_pixels(transform: Affine) -> bool:
    """Return whether the grid's horizontal and vertical pixel sizes are equal."""
    return bool(np.isclose(abs(transform.a), abs(transform.e), rtol=1e-9, atol=1e-9))


def rectangular_grid_profile(profile: dict, transform: Affine, width: int, height: int) -> dict:
    """Create an exact 10 m square-pixel GeoTIFF profile covering a rectangular source grid."""
    source_pixel_x = abs(transform.a)
    source_pixel_y = abs(transform.e)
    target_resolution = RECTANGULAR_PIXEL_OUTPUT_RESOLUTION

    # Preserve the source's upper-left pixel boundary and axis directions.  Rounding up
    # covers the full source footprint, even when it is not an exact multiple of 10 m.
    target_width = math.ceil(width * source_pixel_x / target_resolution)
    target_height = math.ceil(height * source_pixel_y / target_resolution)
    target_transform = Affine(
        math.copysign(target_resolution, transform.a), 0.0, transform.c,
        0.0, math.copysign(target_resolution, transform.e), transform.f,
    )
    target_profile = profile.copy()
    target_profile.update({"width": target_width, "height": target_height, "transform": target_transform})
    return target_profile


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


def find_product_grids(product: h5py.File) -> tuple[str, h5py.Group]:
    """Return the supported product type and its spatial-grid group."""
    matches = [
        (product_type, path)
        for product_type, path in PRODUCT_GRIDS_PATHS.items()
        if path in product
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one supported product; found {matches}")
    product_type, path = matches[0]
    return product_type, product[path]


def dataset_name_for_polarization(product_type: str, polarization: str) -> str:
    """Map a requested channel to its product-specific HDF5 dataset name."""
    if product_type == "GSLC":
        return polarization
    if product_type == "GCOV":
        return polarization * 2
    raise ValueError(f"Unsupported NISAR product type: {product_type}")

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
    # Mark zero, negative, NaN, and infinity values as invalid.  NISAR swath edges
    # commonly contain zero-filled samples, which must not contribute to an average.
    valid = np.isfinite(intensity) & (intensity > 0)
    # Filter data and valid-sample weights separately so NoData never darkens the
    # neighboring valid edge pixels.
    summed = uniform_filter(np.where(valid, intensity, 0.0), size=looks, mode="reflect")
    weights = uniform_filter(valid.astype(np.float32), size=looks, mode="reflect")
    # Preserve invalid source pixels as NaN; they become GeoTIFF NoData on export.
    multilooked = np.full(intensity.shape, np.nan, dtype=np.float32)
    output_valid = valid & (weights > 0)
    multilooked[output_valid] = summed[output_valid] / weights[output_valid]
    return multilooked


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
    # Only positive, finite linear values have a valid dB representation.  Invalid
    # values remain NaN and are written as OUTPUT_NODATA by export_layer, rather than
    # being converted to the artificial -100 dB value produced by log10(1e-10).
    decibels = np.full(intensity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(intensity) & (intensity > 0)
    decibels[valid] = 10.0 * np.log10(intensity[valid])
    return decibels

def export_layer(
    source_file: Path,
    product_type: str,
    frequency: str,
    polarization: str,
    grid: h5py.Group,
    run_timestamp: str,
    logger: logging.Logger,
) -> Path:
    """
    Export one configured GSLC or GCOV polarization layer as a dB GeoTIFF.

    The function reads the source HDF5 dataset tile by tile so a full NISAR scene is
    never loaded into memory. GSLC tiles are complex samples and require intensity,
    multilook, and optional DEM-based RTC processing. GCOV diagonal covariance tiles
    are already intensity-like gamma0 measurements that include multilooking and RTC,
    so they only require dB conversion.

    Args:
        source_file: Input NISAR HDF5/NetCDF4 file used to derive the output name.
        product_type: Detected product identifier, either "GSLC" or "GCOV".
        frequency: Grid frequency group to process, for example "frequencyA".
        polarization: Requested physical channel, for example "HH" or "HV".
        grid: HDF5 group containing coordinate metadata and layer datasets.
        run_timestamp: Shared timestamp included in output filenames for this run.
        logger: Configured logger used for progress, warnings, and diagnostics.

    Raises:
        ValueError: The selected product layer is absent or is not a two-dimensional raster.
        OSError: A source, DEM, temporary GeoTIFF, or output GeoTIFF operation fails.
    """
    # Convert the requested channel name to the dataset name used by this product type.
    dataset_name = dataset_name_for_polarization(product_type, polarization)
    # Keep the HDF5 dataset lazy; tile slices below read only the needed source pixels.
    source_dataset = grid[dataset_name]
    # Reject metadata or unsupported multidimensional datasets before creating an output file.
    if not isinstance(source_dataset, h5py.Dataset) or source_dataset.ndim != 2:
        raise ValueError(f"{dataset_name} is not a two-dimensional raster dataset")

    # Derive the source-native affine transform and CRS from grid coordinate metadata.
    transform, crs = coordinate_transform(grid)
    # Read and process the HDF5 raster at its native dimensions before any output resampling.
    height, width = source_dataset.shape
    # Square source pixels (for example 5 m x 5 m or 10 m x 10 m) are exported unchanged.
    # Rectangular source pixels are converted to a 10 m x 10 m GeoTIFF after processing.
    source_has_square_pixels = grid_has_square_pixels(transform)
    # Record native pixel spacing for DEM slope calculation during GSLC RTC.
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)
    # Only GSLC requires intensity, multilook, and optional DEM correction in this script.
    process_gslc = product_type == "GSLC"

    # Build a distinct filename that records the source, product type, frequency, and channel.
    out_path = PROCESSED_DIRECTORY / (
        f"{source_file.stem}_{product_type}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
    )
    # Write to a temporary file first so an interrupted run cannot leave a partial output.
    temp_path = out_path.with_suffix(".part.tif")
    native_temp_path = out_path.with_suffix(".native.part.tif")
    # Remove temporary files left by a previous interrupted attempt for the same output.
    if temp_path.exists():
        temp_path.unlink()
    if native_temp_path.exists():
        native_temp_path.unlink()

    # Define the native-grid staging GeoTIFF used by both square and rectangular grids.
    profile = {
        "driver": "GTiff",          # Store the result in GeoTIFF format.
        "width": width,              # Preserve the source raster column count in the staging file.
        "height": height,            # Preserve the source raster row count in the staging file.
        "count": 1,                  # Write one dB band per requested channel.
        "dtype": "float32",         # Store dB values as 32-bit floating point values.
        "crs": crs,                  # Retain the grid's coordinate reference system.
        "transform": transform,      # Retain the grid's native affine transform in the staging file.
        "nodata": OUTPUT_NODATA,     # Mark invalid output pixels with a consistent sentinel.
        "tiled": True,               # Enable internal tiles for efficient GIS reading.
        "blockxsize": TILE_SIZE,     # Use the configured tile width for GeoTIFF blocks.
        "blockysize": TILE_SIZE,     # Use the configured tile height for GeoTIFF blocks.
        "compress": "deflate",      # Apply lossless DEFLATE compression.
        "predictor": 3,              # Improve compression efficiency for floating-point data.
        "BIGTIFF": "IF_SAFER",      # Permit BigTIFF output when the file would exceed 4 GB.
    }

    # Start with no DEM handle because GCOV never needs an additional RTC step.
    dem_dataset = None
    # Open the local DEM only for a GSLC run and only when the configured file exists.
    if process_gslc and LOCAL_DEM_PATH.exists():
        dem_dataset = rasterio.open(LOCAL_DEM_PATH)
    # Report that GSLC will continue without RTC when its DEM is unavailable.
    elif process_gslc:
        logger.warning("DEM not found at %s. GSLC RTC step will be skipped.", LOCAL_DEM_PATH)

    if source_has_square_pixels:
        logger.info(
            "%s/%s has square pixels (%.6g m x %.6g m); retaining native resolution.",
            frequency, polarization, pixel_x, pixel_y,
        )
    else:
        logger.info(
            "%s/%s has rectangular pixels (%.6g m x %.6g m); resampling to %.0f m x %.0f m with nearest neighbor.",
            frequency, polarization, pixel_x, pixel_y,
            RECTANGULAR_PIXEL_OUTPUT_RESOLUTION, RECTANGULAR_PIXEL_OUTPUT_RESOLUTION,
        )

    # Count source windows so progress messages can report a meaningful completion percentage.
    total_tiles = ((height + TILE_SIZE - 1) // TILE_SIZE) * ((width + TILE_SIZE - 1) // TILE_SIZE)
    # Describe the actual processing performed in the output raster's band metadata.
    description = "Intensity Multilook RTC" if process_gslc else "RTC Gamma0 Covariance"
    # Reserve a holder for output statistics needed by the QGIS display style.
    band_stats = None

    try:
        # Create the native-grid staging GeoTIFF.
        with rasterio.open(native_temp_path, "w", **profile) as dst:
            # Attach a human-readable label to the sole output band.
            dst.set_band_description(1, f"{product_type} {frequency} {polarization} {description} dB")
            # Iterate over fixed-size source windows to keep memory usage bounded.
            for tile_num, window in enumerate(source_windows(height, width), start=1):
                # Convert Rasterio window offsets to integers accepted by HDF5 slicing.
                row_start = int(window.row_off)
                row_stop = row_start + int(window.height)
                col_start = int(window.col_off)
                col_stop = col_start + int(window.width)
                # Read only this native-grid source tile from the HDF5 dataset.
                tile_data = source_dataset[row_start:row_stop, col_start:col_stop]

                # GSLC source samples are complex-valued and need the GSLC-specific workflow.
                if process_gslc:
                    # Convert complex samples to intensity, then suppress speckle with multilooking.
                    processed_tile = apply_multilook(calculate_intensity(tile_data), looks=5)
                    # Apply DEM-based RTC only when a DEM was opened successfully.
                    if dem_dataset is not None:
                        try:
                            # Build the current tile's affine transform for DEM alignment.
                            win_transform = rasterio.transform.from_bounds(
                                *window_bounds(window, transform), window.width, window.height
                            )
                            # Allocate a DEM array matching the current output tile shape.
                            dem_tile = np.empty((window.height, window.width), dtype=np.float32)
                            # Align the DEM to this tile; this affects only the DEM used by RTC.
                            reproject(
                                source=rasterio.band(dem_dataset, 1),
                                destination=dem_tile,
                                dst_transform=win_transform,
                                dst_crs=crs,
                                resampling=Resampling.bilinear,
                            )
                            # Correct the multilooked GSLC intensity using local terrain slope.
                            processed_tile = apply_rtc(processed_tile, dem_tile, pixel_x, pixel_y)
                        except Exception as exc:
                            # Keep the uncorrected multilooked tile when RTC fails for this window.
                            logger.debug(
                                "RTC failed on tile %d: %s. Using uncorrected intensity.", tile_num, exc
                            )
                # GCOV diagonal covariance values are already multilooked and RTC corrected.
                else:
                    # Convert the native real covariance values to the GeoTIFF float type.
                    processed_tile = np.asarray(tile_data, dtype=np.float32)

                # Convert valid linear intensity or covariance values to the logarithmic dB scale.
                db_tile = convert_to_decibels(processed_tile)
                # Preserve invalid/edge samples as GeoTIFF NoData, not as an in-range
                # dB value.  GIS renderers therefore leave the outside-swath edge blank.
                valid_db = np.isfinite(db_tile)
                db_tile = np.where(valid_db, db_tile, OUTPUT_NODATA)
                # Write this completed native-grid tile into the temporary GeoTIFF.
                dst.write(db_tile.astype(np.float32), 1, window=window)
                # Write an explicit validity mask as well as the NoData value so software
                # that prioritizes raster masks renders the swath edge transparently.
                dst.write_mask(np.where(valid_db, 255, 0).astype(np.uint8), window=window)

                # Log progress approximately ten times per layer and always after the final tile.
                if tile_num == total_tiles or tile_num % max(1, total_tiles // 10) == 0:
                    logger.info(
                        "%s/%s progress: %.0f%% (%d/%d tiles)",
                        frequency,
                        polarization,
                        100 * tile_num / total_tiles,
                        tile_num,
                        total_tiles,
                    )

    finally:
        # Close the DEM even if processing fails while writing a tile.
        if dem_dataset is not None:
            dem_dataset.close()

    if source_has_square_pixels:
        # Square-pixel grids need no resampling, so promote the staging GeoTIFF directly.
        os.replace(native_temp_path, temp_path)
    else:
        # Resample only rectangular grids after the scientific processing steps and before
        # publishing the final GeoTIFF.  Nearest neighbour preserves source dB samples.
        target_profile = rectangular_grid_profile(profile, transform, width, height)
        with rasterio.open(native_temp_path) as source_raster:
            with rasterio.open(temp_path, "w", **target_profile) as destination_raster:
                destination_raster.set_band_description(
                    1, f"{product_type} {frequency} {polarization} {description} dB"
                )
                reproject(
                    source=rasterio.band(source_raster, 1),
                    destination=rasterio.band(destination_raster, 1),
                    src_transform=source_raster.transform,
                    src_crs=source_raster.crs,
                    src_nodata=OUTPUT_NODATA,
                    dst_transform=destination_raster.transform,
                    dst_crs=destination_raster.crs,
                    dst_nodata=OUTPUT_NODATA,
                    resampling=Resampling.nearest,
                    init_dest_nodata=True,
                )
        native_temp_path.unlink()

    # Build overviews on the actual exported grid and calculate its final statistics.
    with rasterio.open(temp_path, "r+") as output_raster:
        output_raster.build_overviews(OVERVIEW_FACTORS, Resampling.average)
        output_raster.update_tags(ns="rio_overview", resampling="average")
        band_stats = output_raster.statistics(1, approx=False)

    # Atomically replace the final output path only after the temporary GeoTIFF is complete.
    os.replace(temp_path, out_path)
    # Record the final product path for operational logs and troubleshooting.
    logger.info("Created processed %s GeoTIFF: %s", product_type, out_path)
    # Return the final GeoTIFF path to the caller.
    return out_path

def main() -> None:
    """
    Discover and process every supported NISAR product below ROOT_DIRECTORY.

    The batch entry point initializes logging, validates input/output directories,
    detects whether each HDF5 file is GSLC or GCOV, and invokes export_layer for
    every configured frequency and polarization that exists in that product. A
    failure for one file is logged and does not stop subsequent files from running.

    Returns:
        None. Outputs and diagnostics are written to configured directories.
    """
    # Configure console and file logging before performing any filesystem work.
    logger = setup_logger(LOG_DIRECTORY)

    # Fail early when the configured input-product directory does not exist.
    if not ROOT_DIRECTORY.is_dir():
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        raise SystemExit(1)

    # Create the output directory now so every successful layer has a destination.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Recursively select supported HDF5 or NetCDF files from the product directory.
    source_files = [
        path
        for path in ROOT_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # End successfully when there are no products to process in the input directory.
    if not source_files:
        logger.warning("No supported products found.")
        return

    # Use one timestamp for every output generated during this batch execution.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # State the amount of batch work before opening the first product.
    logger.info("Found %d product(s). Starting processing...", len(source_files))

    # Process every input file independently so one damaged file cannot stop the batch.
    for source_file in source_files:
        try:
            # Open the HDF5/NetCDF product in read-only mode for metadata and tile access.
            with h5py.File(source_file, "r") as product:
                # Detect the product type and retrieve its product-specific grids group.
                product_type, grids = find_product_grids(product)
                # Record which product type is being processed for operational traceability.
                logger.info("Processing %s product: %s", product_type, source_file.name)

                # Consider each frequency configured at the top of this module.
                for frequency in FREQUENCIES:
                    # Skip a requested frequency that is absent from this product.
                    if frequency not in grids:
                        logger.warning("%s does not contain requested frequency %s.", source_file.name, frequency)
                        continue

                    # Select the grid group holding this frequency's metadata and layers.
                    grid = grids[frequency]
                    # Consider each physical polarization configured at the top of this module.
                    for polarization in POLARIZATIONS:
                        # Convert the requested physical channel to its GSLC or GCOV dataset name.
                        dataset_name = dataset_name_for_polarization(product_type, polarization)
                        # Skip a requested layer that is unavailable in this product acquisition mode.
                        if dataset_name not in grid:
                            logger.warning(
                                "%s %s does not contain requested polarization %s (%s).",
                                source_file.name,
                                frequency,
                                polarization,
                                dataset_name,
                            )
                            continue

                        # Process and export the available layer, resampling only if its pixels are rectangular.
                        export_layer(
                            source_file,
                            product_type,
                            frequency,
                            polarization,
                            grid,
                            run_timestamp,
                            logger,
                        )
        # Handle unreadable or incomplete HDF5 files without stopping the remaining batch.
        except OSError as exc:
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            logger.debug("Truncated file details: %s", exc)
        # Log all other per-product failures with a traceback for later diagnosis.
        except Exception as exc:
            logger.error("Failed to process product %s: %s", source_file.name, exc)
            logger.debug("Detailed traceback:", exc_info=True)

    # Record successful completion after every discoverable input product has been attempted.
    logger.info("Export complete.")

# --- EXECUTION START ---
if __name__ == "__main__":
    main()