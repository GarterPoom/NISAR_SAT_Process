#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import future behavior for modern type hinting support
from __future__ import annotations

# Import math for floating-point pixel-size comparisons
import math

# Import standard library modules for logging system activity
import logging

# Import operating system utilities for file replacement operations
import os

# Import system-specific parameters and functions
import sys

# Import datetime class to format timestamps for filenames and logs
from datetime import datetime

# Import Path class for object-oriented filesystem path operations
from pathlib import Path

# Import Iterator type hint to define return types for generator functions
from typing import Iterator

# Import h5py to read HDF5 and NetCDF4 scientific data files
import h5py

# Import numpy for array manipulation and mathematical operations
import numpy as np

# Import uniform_filter from scipy.ndimage for multi-looking (spatial averaging)
from scipy.ndimage import uniform_filter

# Import rasterio to read DEMs and write geospatial raster data (GeoTIFFs)
import rasterio

# Import Resampling enum from rasterio to define pyramid overview resampling algorithms
from rasterio.enums import Resampling

# Import reprojection utilities to resample non-square pixels to a uniform grid
from rasterio.warp import calculate_default_transform, reproject

# Import array_bounds to compute spatial extent from a transform
from rasterio.transform import array_bounds

# Import Affine to construct coordinate transform matrices for georeferencing
from affine import Affine

# Import Window to define pixel sub-region tiles for memory-efficient writing
from rasterio.windows import Window

from rasterio.windows import bounds as window_bounds


# Determine the absolute directory path where this script file is located
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Define the root input directory where HDF5 and NetCDF files are stored
ROOT_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_Product"

# Define the target directory where the final dB processed GeoTIFF files will be saved
PROCESSED_DIRECTORY = SCRIPT_DIRECTORY / "GeoTIFF_Processed"

# Define the directory where runtime execution log files will be saved
LOG_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_logs"

# Define the absolute path to your Local DEM used for Radiometric Terrain Correction
# WARNING: Ensure this DEM is co-registered and matches the output CRS!
LOCAL_DEM_PATH = SCRIPT_DIRECTORY / "NASA_DEM" / "NISAR_DEM_1-20260817_064201_Mosaic.tif"

# Define the internal HDF5/NetCDF path hierarchy leading to GSLC spatial grids
GSLC_GRIDS_PATH = "science/LSAR/GSLC/grids"

# Define the tuple of sub-band frequency sub-groups to extract
FREQUENCIES = ("frequencyA",)

# Define the tuple of polarization channel datasets to extract
POLARIZATIONS = ("HH",)

# Define the tuple of supported file extensions (HDF5 and NetCDF4)
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

# Set the tile block size (512x512 pixels) for memory-efficient chunked processing
TILE_SIZE = 512

# Define default downsampling overview pyramid factors
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

# Define the target square pixel size (meters) that non-square source pixels are resampled to
TARGET_PIXEL_SIZE = 10.0


def setup_logger(log_directory: Path) -> logging.Logger:
    """Write DEBUG-level details to both the terminal and a dynamic log file."""
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"NISAR_L_Band_Process_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("NISAR_L_Band_Process")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.info("Log file: %s", log_path)
    logger.info("Detailed terminal logging is enabled.")
    return logger


def coordinate_transform(grid: h5py.Group) -> tuple[Affine, str]:
    """Build pixel-corner GeoTIFF georeferencing from NISAR grid coordinates."""
    x_coordinates = grid["xCoordinates"]
    y_coordinates = grid["yCoordinates"]
    
    if len(x_coordinates) < 2 or len(y_coordinates) < 2:
        raise ValueError("xCoordinates and yCoordinates must each contain at least two values")
        
    x_resolution = float(x_coordinates[1] - x_coordinates[0])
    y_resolution = float(y_coordinates[1] - y_coordinates[0])
    
    if x_resolution == 0 or y_resolution == 0:
        raise ValueError("Coordinate spacing cannot be zero")
        
    projection = grid["projection"]
    epsg_code = int(projection.attrs.get("epsg_code", projection[()]))
    
    transform = Affine(x_resolution, 0.0, float(x_coordinates[0]) - x_resolution / 2,
                       0.0, y_resolution, float(y_coordinates[0]) - y_resolution / 2)
    return transform, f"EPSG:{epsg_code}"


def source_windows(height: int, width: int) -> Iterator[Window]:
    """Yield source-sized tiles so very large rasters are never loaded at once."""
    for row_offset in range(0, height, TILE_SIZE):
        tile_height = min(TILE_SIZE, height - row_offset)
        for column_offset in range(0, width, TILE_SIZE):
            tile_width = min(TILE_SIZE, width - column_offset)
            yield Window(column_offset, row_offset, tile_width, tile_height)


def calculate_intensity(complex_data: np.ndarray) -> np.ndarray:
    """Calculate Intensity from complex GSLC data (Intensity = Real^2 + Imaginary^2)."""
    real_sq = np.square(np.real(complex_data)).astype(np.float32)
    imag_sq = np.square(np.imag(complex_data)).astype(np.float32)
    return real_sq + imag_sq


def apply_multilook(intensity: np.ndarray, looks: int = 5) -> np.ndarray:
    """Perform Multi-looking speckle filtering using a spatial average."""
    return uniform_filter(intensity, size=looks, mode="reflect")


def apply_rtc(intensity: np.ndarray, dem_array: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """Apply basic Radiometric Terrain Correction using DEM slope."""
    if dem_array.shape[0] < 2 or dem_array.shape[1] < 2:
        return intensity

    dy, dx = np.gradient(dem_array, res_y, res_x)
    slope = np.sqrt(dx**2 + dy**2)
    cos_i = np.cos(np.arctan(slope)) + 1e-6
    return intensity / cos_i


def convert_to_decibels(intensity: np.ndarray) -> np.ndarray:
    """Convert Intensity to Decibels (dB) by 10 * log10(Intensity)."""
    safe_intensity = np.where(intensity > 0, intensity, 1e-10)
    return 10.0 * np.log10(safe_intensity)


def export_layer(
    source_file: Path, frequency: str, polarization: str, grid: h5py.Group, run_timestamp: str, logger: logging.Logger
) -> Path:
    """Read complex data, calculate Intensity, Multilook, Apply RTC, convert to dB, and export raster."""
    source_dataset = grid[polarization]
    transform, crs = coordinate_transform(grid)
    height, width = source_dataset.shape
    is_complex = source_dataset.dtype.kind == "c"

    pixel_x, pixel_y = abs(transform.a), abs(transform.e)
    needs_resample = not math.isclose(pixel_x, pixel_y, rel_tol=1e-9, abs_tol=1e-6)

    if needs_resample:
        logger.info("Resampling to %.0f x %.0f m.", TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE)
        full_array = source_dataset[:, :]
        
        if is_complex:
            full_array = calculate_intensity(full_array)
            is_complex = False
            
        left, bottom, right, top = array_bounds(height, width, transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, crs, width, height, left, bottom, right, top, resolution=(TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE)
        )
        
        resampled_array = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=full_array, destination=resampled_array,
            src_transform=transform, src_crs=crs,
            dst_transform=dst_transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
        source_dataset = resampled_array
        transform = dst_transform
        height, width = dst_height, dst_width

    out_filename = f"{source_file.stem}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
    out_path = PROCESSED_DIRECTORY / out_filename
    temp_path = out_path.with_suffix(".part.tif")
    
    if temp_path.exists():
        temp_path.unlink()

    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": -9999.0,
        "tiled": True, "blockxsize": TILE_SIZE, "blockysize": TILE_SIZE,
        "compress": "deflate", "predictor": 3, "BIGTIFF": "IF_SAFER",
    }

    dem_dataset = None
    if LOCAL_DEM_PATH.exists():
        dem_dataset = rasterio.open(LOCAL_DEM_PATH)
    else:
        logger.warning(f"DEM not found at {LOCAL_DEM_PATH}. RTC step will be skipped.")

    total_tiles = ((height + TILE_SIZE - 1) // TILE_SIZE) * ((width + TILE_SIZE - 1) // TILE_SIZE)

    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.set_band_description(1, f"{frequency} {polarization} Intensity Multilook RTC dB")

        for tile_num, window in enumerate(source_windows(height, width), start=1):
            tile_data = source_dataset[window.row_off:window.row_off+window.height, 
                                       window.col_off:window.col_off+window.width]
            
            if is_complex:
                intensity_tile = calculate_intensity(tile_data)
            else:
                intensity_tile = tile_data

            ml_tile = apply_multilook(intensity_tile, looks=5)

            if dem_dataset:
                try:
                    win_bounds = window_bounds(window, transform)
                    win_transform = rasterio.transform.from_bounds(*win_bounds, window.width, window.height)
                    dem_tile = np.empty((window.height, window.width), dtype=np.float32)

                    reproject(
                        source=rasterio.band(dem_dataset, 1),
                        destination=dem_tile,
                        dst_transform=win_transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear
                    )
                    rtc_tile = apply_rtc(ml_tile, dem_tile, abs(transform.a), abs(transform.e))
                except Exception as e:
                    logger.debug(f"RTC failed on tile {tile_num}: {e}. Using uncorrected intensity.")
                    rtc_tile = ml_tile
            else:
                rtc_tile = ml_tile

            db_tile = convert_to_decibels(rtc_tile)
            db_tile = np.where(np.isfinite(db_tile), db_tile, -9999.0)
            dst.write(db_tile.astype(np.float32), 1, window=window)

            if tile_num == total_tiles or tile_num % max(1, total_tiles // 10) == 0:
                logger.info("%s/%s progress: %.0f%% (%d/%d tiles)", frequency, polarization, 100 * tile_num / total_tiles, tile_num, total_tiles)

        dst.build_overviews(OVERVIEW_FACTORS, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    if dem_dataset:
        dem_dataset.close()

    os.replace(temp_path, out_path)
    logger.info("Created Processed dB GeoTIFF: %s", out_path)
    return out_path


def main() -> None:
    """Coordinate the processing and export of requested GSLC datasets."""
    logger = setup_logger(LOG_DIRECTORY)
    
    if not ROOT_DIRECTORY.is_dir():
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        raise SystemExit(1)
        
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    source_files = [p for p in ROOT_DIRECTORY.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not source_files:
        logger.warning("No supported products found.")
        return
        
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Found %d product(s). Starting processing...", len(source_files))

    for source_file in source_files:
        try:
            with h5py.File(source_file, "r") as product:
                grids = product[GSLC_GRIDS_PATH]
                for frequency in FREQUENCIES:
                    grid = grids[frequency]
                    for polarization in POLARIZATIONS:
                        export_layer(source_file, frequency, polarization, grid, run_timestamp, logger)
        
        # ADDED: Catching the specific OSError that happens with truncated/corrupted downloads
        except OSError as e:
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            logger.debug("Truncated file details: %s", e)
            
        # Standard exception block for other unexpected errors
        except Exception as e:
            logger.error("Failed to process product %s: %s", source_file.name, e)
            logger.debug("Detailed traceback:", exc_info=True)

    logger.info("Export complete.")


if __name__ == "__main__":
    main()