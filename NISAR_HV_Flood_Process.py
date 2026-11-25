#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NISAR_HV_Flood_Process.py

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

# Import re to extract coordinates from the WKT footprint string in the product metadata
import re

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
ROOT_DIRECTORY = Path(r"NISAR_Product") #SCRIPT_DIRECTORY / "NISAR_Product"

# Directory where the processed, georeferenced GeoTIFF files will be saved
PROCESSED_DIRECTORY = SCRIPT_DIRECTORY / "GeoTIFF_Processed"

# Separate directory where binary flood-mask GeoTIFF files will be saved
FLOOD_DIRECTORY = SCRIPT_DIRECTORY / "Flood_Raster"

# Directory for storing execution logs for debugging and auditing
LOG_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_logs"

# Path to the Digital Elevation Model (DEM) used for Terrain Correction (RTC)
# This must match the CRS and spatial resolution of your target area
LOCAL_DEM_PATH = SCRIPT_DIRECTORY / "NASA_DEM" / "NISAR_DEM_1-20260817_064201_Mosaic.tif"

# Internal HDF5 paths for the supported NISAR products.
PRODUCT_GRIDS_PATHS = {
    "GSLC": "science/LSAR/GSLC/grids",
    "GCOV": "science/LSAR/GCOV/grids",
}

# Tuple specifying which frequency band to process (e.g., L-Band frequencyA)
FREQUENCIES = ("frequencyA",)

# Tuple specifying which polarization channels to extract (e.g., HH, HV polarization)
POLARIZATIONS = ("HV",)

# A list of file extensions that the script will recognize as valid input files
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

# The dimensions (height/width) of the tiles used during processing to prevent memory overflow
TILE_SIZE = 512

# GeoTIFF value used for pixels outside the valid NISAR swath.  This is deliberately
# outside the expected backscatter dB range, so invalid edge pixels are transparent
# in GIS software instead of appearing as a black -100 dB border.
OUTPUT_NODATA = -9999.0

# Fixed dB threshold for the flood mask. Pixels strictly below this value are
# flood (1); pixels equal to or above -20 dB and source NoData are non-flood (0).
FLOOD_THRESHOLD_DB = -20.0

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
        # Let each handler pass every message from DEBUG level upward.
        handler.setLevel(logging.DEBUG)
        # Apply the shared timestamp format to this handler.
        handler.setFormatter(formatter)
        # Attach the handler so messages go to the terminal and the log file.
        logger.addHandler(handler)

    # Tell the user where the log file is written.
    logger.info("Log file: %s", log_path)
    # Confirm that detailed logging is active.
    logger.info("Detailed terminal logging is enabled.")
    # Return the configured logger to the caller.
    return logger


# Build the GeoTIFF affine transform and CRS from the HDF5 grid coordinates.
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
    # Northing (y) coordinate of every pixel row.
    y_coordinates = grid["yCoordinates"]

    # Validation: We need at least two points to calculate resolution (distance between points)
    if len(x_coordinates) < 2 or len(y_coordinates) < 2:
        # Stop: pixel spacing cannot be computed from fewer than two coordinates.
        raise ValueError("xCoordinates and yCoordinates must each contain at least two values")

    # Calculate pixel resolution by finding the difference between adjacent coordinates
    x_resolution = float(x_coordinates[1] - x_coordinates[0])
    # Pixel height (negative when rows run north to south).
    y_resolution = float(y_coordinates[1] - y_coordinates[0])

    # Prevent division by zero errors
    if x_resolution == 0 or y_resolution == 0:
        # Stop: zero spacing would make an invalid transform.
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

    # Return the transform and the CRS as an "EPSG:<code>" string.
    return transform, f"EPSG:{epsg_code}"


# Check whether the pixel width equals the pixel height.
def grid_has_square_pixels(transform: Affine) -> bool:
    """Return whether the grid's horizontal and vertical pixel sizes are equal."""
    # Compare absolute pixel width (a) and height (e) with a tiny tolerance.
    return bool(np.isclose(abs(transform.a), abs(transform.e), rtol=1e-9, atol=1e-9))


# Build the output profile that resamples a rectangular-pixel grid to square pixels.
def rectangular_grid_profile(profile: dict, transform: Affine, width: int, height: int) -> dict:
    """Create an exact 10 m square-pixel GeoTIFF profile covering a rectangular source grid."""
    # Source pixel width in metres.
    source_pixel_x = abs(transform.a)
    # Source pixel height in metres.
    source_pixel_y = abs(transform.e)
    # Output square pixel size in metres.
    target_resolution = RECTANGULAR_PIXEL_OUTPUT_RESOLUTION

    # Preserve the source's upper-left pixel boundary and axis directions.  Rounding up
    # covers the full source footprint, even when it is not an exact multiple of 10 m.
    target_width = math.ceil(width * source_pixel_x / target_resolution)
    # Number of output rows needed to cover the source height.
    target_height = math.ceil(height * source_pixel_y / target_resolution)
    # Output transform: same upper-left corner and axis directions, new pixel size.
    target_transform = Affine(
        math.copysign(target_resolution, transform.a), 0.0, transform.c,
        0.0, math.copysign(target_resolution, transform.e), transform.f,
    )
    # Copy the source profile so the original is left unchanged.
    target_profile = profile.copy()
    # Apply the new grid size and transform.
    target_profile.update({"width": target_width, "height": target_height, "transform": target_transform})
    # Return the profile for the square-pixel output.
    return target_profile


# Yield TILE_SIZE x TILE_SIZE windows that together cover the whole raster.
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


# Detect whether the file is GSLC or GCOV and return its grids group.
def find_product_grids(product: h5py.File) -> tuple[str, h5py.Group]:
    """Return the supported product type and its spatial-grid group."""
    # List every supported product type whose grids path exists in this file.
    matches = [
        (product_type, path)
        for product_type, path in PRODUCT_GRIDS_PATHS.items()
        if path in product
    ]
    # Exactly one product type must match.
    if len(matches) != 1:
        # Stop: the file is not a single GSLC or GCOV product.
        raise ValueError(f"Expected exactly one supported product; found {matches}")
    # Unpack the only match.
    product_type, path = matches[0]
    # Return the product type and its HDF5 grids group.
    return product_type, product[path]


# Translate a polarization (HH/HV) into the HDF5 dataset name for this product type.
def dataset_name_for_polarization(product_type: str, polarization: str) -> str:
    """Map a requested channel to its product-specific HDF5 dataset name."""
    # GSLC product.
    if product_type == "GSLC":
        # GSLC stores each channel under its own name, e.g. "HH".
        return polarization
    # GCOV product.
    if product_type == "GCOV":
        # GCOV stores diagonal covariance terms, e.g. "HHHH".
        return polarization * 2
    # Any other product type is not supported.
    raise ValueError(f"Unsupported NISAR product type: {product_type}")

# --- SOURCE METADATA LOGGING ---

# Root-level HDF5 attributes that identify the product family and its governing specification.
ROOT_ATTRIBUTES = ("title", "mission_name", "institution", "contact", "reference_document", "Conventions")

# Per-frequency datasets under science/LSAR/<product>/grids/<frequency> worth recording.
GRID_FREQUENCY_KEYS = (
    "listOfPolarizations", "centerFrequency", "rangeBandwidth", "azimuthBandwidth",
    "xCoordinateSpacing", "yCoordinateSpacing", "projection", "numberOfSubSwaths",
)

# Per-frequency acquisition geometry under metadata/sourceData/swaths/<frequency>.
SWATH_FREQUENCY_KEYS = (
    "acquiredRangeBandwidth", "processedRangeBandwidth", "processedAzimuthBandwidth",
    "nearRangeIncidenceAngle", "farRangeIncidenceAngle", "rangeResolution",
    "sceneCenterAlongTrackResolution", "sceneCenterGroundRangeSpacing",
    "sceneCenterAlongTrackSpacing", "slantRangeStart", "slantRangeSpacing",
    "numberOfRangeSamples",
)

# Processing-parameter flags that state which corrections were already applied upstream.
PROCESSING_PARAMETER_KEYS = (
    "azimuthIonosphericGeolocationCorrectionApplied", "rangeIonosphericGeolocationCorrectionApplied",
    "dryTroposphericGeolocationCorrectionApplied", "wetTroposphericGeolocationCorrectionApplied",
    "ellipsoidalFlatteningApplied", "topographicFlatteningApplied",
)

# Largest dataset (in elements) that is read into memory as metadata. Raster layers such as
# HH (billions of samples) are far above this and are only described by shape and dtype.
MAX_METADATA_ELEMENTS = 16

# Longest string logged on a single line; larger blobs (run configuration, WKT) are summarised.
MAX_METADATA_STRING = 300


# Turn an HDF5/numpy value into a plain Python value that prints cleanly.
def to_plain_value(value):
    """Convert h5py/numpy values (bytes, scalars, arrays) to plain Python types for logging."""
    # HDF5 strings are often stored as bytes.
    if isinstance(value, bytes):
        # Decode bytes to text, replacing any invalid characters.
        return value.decode("utf-8", errors="replace")
    # Arrays are converted element by element.
    if isinstance(value, np.ndarray):
        # Return a Python list of plain values.
        return [to_plain_value(item) for item in value.tolist()]
    # numpy scalar types (e.g. np.float64).
    if isinstance(value, np.generic):
        # Convert to the equivalent Python scalar.
        return to_plain_value(value.item())
    # Already a plain Python value.
    return value


# Read a small metadata dataset and its units, skipping large raster layers.
def read_metadata_dataset(group: h5py.Group, name: str):
    """Return ``(value, units)`` for a small dataset, or None when absent or too large.

    Only scalar or tiny datasets are read, so a rasterised layer can never be loaded here.
    """
    # The dataset does not exist in this group.
    if name not in group:
        # Nothing to read.
        return None
    # Get the HDF5 object stored under this name.
    dataset = group[name]
    # Skip sub-groups and datasets too large to be metadata.
    if not isinstance(dataset, h5py.Dataset) or dataset.size > MAX_METADATA_ELEMENTS:
        # Not metadata, so return nothing.
        return None
    # Read the dataset's units attribute, if any.
    units = to_plain_value(dataset.attrs.get("units", ""))
    # Dimensionless markers ("unitless", "1") add noise to the log, so they are dropped.
    if units in ("unitless", "1"):
        # Treat the dimensionless marker as having no units.
        units = ""
    # Return the dataset value and its units.
    return to_plain_value(dataset[()]), units


# Write one aligned "label : value [units]" line to the log.
def log_metadata_value(logger: logging.Logger, label: str, value, units: str = "") -> None:
    """Write one ``label: value [units]`` line, summarising oversized strings."""
    # Very long strings would flood the log.
    if isinstance(value, str) and len(value) > MAX_METADATA_STRING:
        # Replace them with a short placeholder giving their length.
        value = f"<{len(value)} characters omitted>"
    # Append the units in brackets when they are known.
    suffix = f" [{units}]" if units else ""
    # Write the line with the label padded to a fixed width.
    logger.info("  %-44s : %s%s", label, value, suffix)


# Log the small datasets of one HDF5 group (all of them, or only the given keys).
def log_group_metadata(
    logger: logging.Logger,
    product: h5py.File,
    group_path: str,
    keys: tuple[str, ...] | None = None,
) -> None:
    """Log selected (or, with ``keys=None``, every small) datasets of one HDF5 group.

    Missing groups or keys are skipped silently because the metadata layout differs between
    GSLC and GCOV products and between processing-software versions.
    """
    # The group does not exist in this product.
    if group_path not in product:
        # Nothing to log.
        return
    # Open the HDF5 group.
    group = product[group_path]
    # Use the requested keys, or every key in the group in sorted order.
    for name in (keys if keys is not None else sorted(group.keys())):
        # Read the value and units, or None when missing or too large.
        found = read_metadata_dataset(group, name)
        # Log only datasets that were actually read.
        if found is not None:
            # Write the "name : value [units]" line.
            log_metadata_value(logger, name, *found)


# Log the product footprint as a longitude/latitude range.
def log_bounding_polygon(logger: logging.Logger, product: h5py.File) -> None:
    """Log the ``boundingPolygon`` WKT footprint as a longitude/latitude bounding box."""
    # HDF5 group that holds the footprint polygon.
    identification_path = "science/LSAR/identification"
    # Nothing to log when the footprint is missing.
    if identification_path not in product or "boundingPolygon" not in product[identification_path]:
        # Nothing to log.
        return
    # Read the footprint WKT text.
    wkt = to_plain_value(product[identification_path]["boundingPolygon"][()])
    # WKT vertices are "lon lat height" triples; the height component is not needed here.
    numbers = np.array(re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", wkt), dtype=float)
    # Need at least two complete (lon, lat, height) vertices.
    if numbers.size < 6 or numbers.size % 3:
        # Malformed footprint, so skip it.
        return
    # Take every third number as longitude and the next as latitude.
    longitudes, latitudes = numbers[0::3], numbers[1::3]
    # Log the number of footprint vertices.
    log_metadata_value(logger, "footprintVertices", len(longitudes))
    # Log the west-to-east extent.
    log_metadata_value(logger, "footprintLongitudeRange", f"{longitudes.min():.5f} to {longitudes.max():.5f}", "degrees")
    # Log the south-to-north extent.
    log_metadata_value(logger, "footprintLatitudeRange", f"{latitudes.min():.5f} to {latitudes.max():.5f}", "degrees")


# Log the source file's identification, orbit, processing and grid metadata.
def log_product_metadata(
    product: h5py.File,
    source_file: Path,
    product_type: str,
    grids: h5py.Group,
    logger: logging.Logger,
) -> None:
    """Log the original file's metadata before any of its layers are processed.

    Only metadata is read (attributes and tiny datasets); the raster layers are described by
    shape and dtype without loading pixels. Because the work happens before processing, the
    record is written even when every layer is later skipped as already completed.
    """
    # HDF5 root of this product's science data.
    product_root = f"science/LSAR/{product_type}"
    # Draw a separator line in the log.
    logger.info("=" * 78)
    # Header naming the source file.
    logger.info("SOURCE METADATA: %s", source_file.name)
    # Draw a separator line in the log.
    logger.info("=" * 78)
    # Read metadata defensively so a layout surprise never stops processing.
    try:
        # Section: file information.
        logger.info("[File]")
        # Log the full path of the source file.
        log_metadata_value(logger, "path", source_file.resolve())
        # Log the file size in GiB.
        log_metadata_value(logger, "sizeOnDisk", f"{source_file.stat().st_size / 1024 ** 3:.3f}", "GiB")
        # Log the root-level descriptive attributes.
        for name in ROOT_ATTRIBUTES:
            # Log only attributes that this file has.
            if name in product.attrs:
                # Write the attribute value.
                log_metadata_value(logger, name, to_plain_value(product.attrs[name]))

        # Section: product identification.
        logger.info("[Identification]")
        # Log every small identification dataset.
        log_group_metadata(logger, product, "science/LSAR/identification")
        # Log the footprint extent.
        log_bounding_polygon(logger, product)

        # Section: orbit information.
        logger.info("[Orbit]")
        # Log the orbit type and interpolation method.
        log_group_metadata(logger, product, f"{product_root}/metadata/orbit", ("orbitType", "interpMethod"))
        # Orbit state vectors are present.
        if f"{product_root}/metadata/orbit/position" in product:
            # Log how many orbit state vectors the product contains.
            log_metadata_value(
                logger, "orbitStateVectors", product[f"{product_root}/metadata/orbit/position"].shape[0]
            )

        # Section: corrections applied by the NISAR processor.
        logger.info("[Processing Parameters]")
        # Log the selected datasets of this metadata group.
        log_group_metadata(
            logger, product, f"{product_root}/metadata/processingInformation/parameters",
            PROCESSING_PARAMETER_KEYS,
        )

        # Repeat for every configured frequency band.
        for frequency in FREQUENCIES:
            # This frequency is not in the product.
            if frequency not in grids:
                # Skip it.
                continue
            # HDF5 group of this frequency's grid.
            grid = grids[frequency]
            # Section: grid metadata for this frequency.
            logger.info("[Grid %s]", frequency)
            # Log the selected grid datasets.
            log_group_metadata(logger, product, grid.name.lstrip("/"), GRID_FREQUENCY_KEYS)
            # Repeat for every configured polarization.
            for polarization in POLARIZATIONS:
                # HDF5 dataset name of this polarization layer.
                dataset_name = dataset_name_for_polarization(product_type, polarization)
                # Only describe layers that exist.
                if dataset_name in grid:
                    # Open the layer without reading its pixels.
                    layer = grid[dataset_name]
                    # Log the layer size and data type.
                    log_metadata_value(logger, f"layer {dataset_name}", f"shape={layer.shape} dtype={layer.dtype}")
            # Section: acquisition geometry for this frequency.
            logger.info("[Swath %s]", frequency)
            # Log the selected acquisition-geometry datasets.
            log_group_metadata(
                logger, product, f"{product_root}/metadata/sourceData/swaths/{frequency}", SWATH_FREQUENCY_KEYS
            )
    except Exception as exc:
        # Metadata is informational; a layout surprise must never stop the product's processing.
        logger.warning("Could not read all source metadata for %s: %s", source_file.name, exc)
        # Keep the full traceback in the debug log.
        logger.debug("Metadata traceback:", exc_info=True)
    # Draw a separator line in the log.
    logger.info("=" * 78)


# Convert complex SAR samples to intensity (|z|^2).
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


# Average intensity over a looks x looks window to reduce speckle, ignoring NoData.
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
    # Fraction of valid pixels in each averaging window.
    weights = uniform_filter(valid.astype(np.float32), size=looks, mode="reflect")
    # Preserve invalid source pixels as NaN; they become GeoTIFF NoData on export.
    multilooked = np.full(intensity.shape, np.nan, dtype=np.float32)
    # Pixels that are valid and have at least one valid neighbour.
    output_valid = valid & (weights > 0)
    # Mean of the valid pixels only.
    multilooked[output_valid] = summed[output_valid] / weights[output_valid]
    # Return the speckle-reduced intensity.
    return multilooked


# Simple slope-based radiometric terrain correction using a DEM tile.
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
        # DEM tile too small for a gradient, so leave intensity unchanged.
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


# Convert linear intensity to decibels, leaving invalid values as NaN.
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
    # Only positive, finite values have a dB value.
    valid = np.isfinite(intensity) & (intensity > 0)
    # dB = 10 * log10(intensity).
    decibels[valid] = 10.0 * np.log10(intensity[valid])
    # Return the dB array.
    return decibels


# Find a finished GeoTIFF from an earlier run for this layer, if one exists.
def completed_output_path(
    source_file: Path,
    product_type: str,
    frequency: str,
    polarization: str,
    logger: logging.Logger,
) -> Path | None:
    """Return an existing final GeoTIFF for a layer, ignoring partial outputs.

    Output filenames include the batch timestamp, so search for any prior final
    export of this source/product/frequency/polarization combination.  The final
    GeoTIFF is only published with ``os.replace`` after writing, overviews, and
    statistics all succeed; ``*.part.tif`` files are therefore never treated as
    completed results.
    """
    # Filename prefix shared by every export of this layer, regardless of run timestamp.
    output_prefix = (
        f"{source_file.stem}_{product_type}_{frequency}_{polarization}_Processed_dB_"
    )
    # Matching GeoTIFFs, newest first.
    candidates = sorted(
        PROCESSED_DIRECTORY.glob(f"{output_prefix}*.tif"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        # Interrupted exports retain a .part.tif suffix and must be retried.
        if candidate.name.endswith(".part.tif"):
            # Skip interrupted outputs.
            continue
        try:
            # Confirm the final-named file can still be opened as a nonempty GeoTIFF.
            with rasterio.open(candidate) as existing_raster:
                # Accept the file only if it is a non-empty single-band GeoTIFF.
                if (
                    existing_raster.driver == "GTiff"
                    and existing_raster.count == 1
                    and existing_raster.width > 0
                    and existing_raster.height > 0
                ):
                    # Reuse this finished GeoTIFF.
                    return candidate
        except (OSError, rasterio.errors.RasterioError) as exc:
            # A damaged result is not complete and should be regenerated.
            logger.warning("Existing GeoTIFF is unreadable and will be regenerated: %s (%s)", candidate, exc)

    # No finished GeoTIFF was found.
    return None


# Build the flood mask filename for a processed dB GeoTIFF.
def flood_raster_path(processed_path: Path) -> Path:
    """Return the companion flood-mask filename for a processed dB GeoTIFF."""
    # Flood mask path: source stem plus the threshold used.
    return FLOOD_DIRECTORY / (
        f"{processed_path.stem}_Flood_Raster_lt_{FLOOD_THRESHOLD_DB:g}dB.tif"
    )


# Threshold a processed dB GeoTIFF into a binary flood mask (1 = flood).
def export_flood_raster(processed_path: Path, logger: logging.Logger) -> Path:
    """Create a binary flood raster from a processed dB GeoTIFF.

    Flood pixels are encoded as ``1`` where the dB value is strictly less than
    ``FLOOD_THRESHOLD_DB``. Every other pixel, including source NoData, is encoded
    as ``0`` so the output contains only the binary values 0 and 1.
    """
    # Final flood mask path.
    output_path = flood_raster_path(processed_path)
    # Make sure the flood output folder exists.
    FLOOD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Temporary file used until the mask is complete.
    temp_path = output_path.with_suffix(".part.tif")
    # A temporary file from an earlier run is present.
    if temp_path.exists():
        # Delete the leftover temporary file.
        temp_path.unlink()

    # Open the processed dB GeoTIFF.
    with rasterio.open(processed_path) as source_raster:
        # Start from the source raster's layout and georeferencing.
        profile = source_raster.profile.copy()
        # Switch to a compressed single-band uint8 output with no NoData value.
        profile.update(
            {
                "dtype": "uint8",
                "count": 1,
                "nodata": None,
                "compress": "deflate",
                "predictor": 2,
                "BIGTIFF": "IF_SAFER",
            }
        )
        # Create the temporary flood mask GeoTIFF.
        with rasterio.open(temp_path, "w", **profile) as flood_raster:
            # Describe the band, including the threshold.
            flood_raster.set_band_description(
                1, f"Flood mask: dB < {FLOOD_THRESHOLD_DB:g}"
            )
            # Record the threshold rule and pixel meanings as metadata tags.
            flood_raster.update_tags(
                flood_threshold_db=str(FLOOD_THRESHOLD_DB),
                threshold_operator="<",
                flood_pixel_value="1",
                non_flood_pixel_value="0",
            )

            # Reuse the processed raster's block layout to keep memory usage bounded.
            for _, window in source_raster.block_windows(1):
                # Read one block of dB values with NoData masked.
                db_tile = source_raster.read(1, window=window, masked=True)
                # Flood where the pixel is valid, finite and below the threshold.
                flood_pixels = (
                    ~np.ma.getmaskarray(db_tile)
                    & np.isfinite(db_tile.data)
                    & (db_tile.data < FLOOD_THRESHOLD_DB)
                )
                # Store True/False as 1/0.
                flood_tile = flood_pixels.astype(np.uint8)
                # Write the block into the mask.
                flood_raster.write(flood_tile, 1, window=window)

            # Do not request duplicate 1x1 overviews for small rasters.
            valid_overview_factors = [
                factor
                for factor in OVERVIEW_FACTORS
                if factor <= min(flood_raster.width, flood_raster.height)
            ]
            # Build overviews only when at least one factor fits the raster.
            if valid_overview_factors:
                # Nearest-neighbour overviews keep the 0/1 values.
                flood_raster.build_overviews(valid_overview_factors, Resampling.nearest)
                # Record the overview resampling method.
                flood_raster.update_tags(ns="rio_overview", resampling="nearest")

    # Publish the completed raster atomically and replace an older mask of the same run.
    os.replace(temp_path, output_path)
    # Report the written flood mask and the threshold used.
    logger.info(
        "Created binary flood raster (dB < %g): %s", FLOOD_THRESHOLD_DB, output_path
    )
    # Return the path of the flood mask.
    return output_path

# Process one frequency/polarization layer into a dB GeoTIFF.
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
    # Avoid repeating expensive tile processing when a previous run published this layer.
    existing_output = completed_output_path(
        source_file, product_type, frequency, polarization, logger
    )
    # A finished GeoTIFF for this layer already exists.
    if existing_output is not None:
        # Report that this layer is skipped and where the existing file is.
        logger.info(
            "Skipping %s/%s; completed GeoTIFF already exists: %s",
            frequency,
            polarization,
            existing_output,
        )
        # Reuse it instead of processing again.
        return existing_output

    # Convert the requested channel name to the dataset name used by this product type.
    dataset_name = dataset_name_for_polarization(product_type, polarization)
    # Keep the HDF5 dataset lazy; tile slices below read only the needed source pixels.
    source_dataset = grid[dataset_name]
    # Reject metadata or unsupported multidimensional datasets before creating an output file.
    if not isinstance(source_dataset, h5py.Dataset) or source_dataset.ndim != 2:
        # Stop: this dataset is not a 2-D raster.
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
    # Native pixel height in metres.
    pixel_y = abs(transform.e)
    # Only GSLC requires intensity, multilook, and optional DEM correction in this script.
    process_gslc = product_type == "GSLC"

    # Build a distinct filename that records the source, product type, frequency, and channel.
    out_path = PROCESSED_DIRECTORY / (
        f"{source_file.stem}_{product_type}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
    )
    # Write to a temporary file first so an interrupted run cannot leave a partial output.
    temp_path = out_path.with_suffix(".part.tif")
    # Temporary native-grid file used before any resampling.
    native_temp_path = out_path.with_suffix(".native.part.tif")
    # Remove temporary files left by a previous interrupted attempt for the same output.
    if temp_path.exists():
        # Delete the leftover temporary file.
        temp_path.unlink()
    # A native-grid temporary file from an earlier run is present.
    if native_temp_path.exists():
        # Delete the leftover native-grid temporary file.
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
        # Open the DEM for terrain correction.
        dem_dataset = rasterio.open(LOCAL_DEM_PATH)
    # Report that GSLC will continue without RTC when its DEM is unavailable.
    elif process_gslc:
        # Warn that terrain correction is skipped.
        logger.warning("DEM not found at %s. GSLC RTC step will be skipped.", LOCAL_DEM_PATH)

    # Square pixels are written at native resolution.
    if source_has_square_pixels:
        # Report that the native resolution is kept.
        logger.info(
            "%s/%s has square pixels (%.6g m x %.6g m); retaining native resolution.",
            frequency, polarization, pixel_x, pixel_y,
        )
    else:
        # Report that the rectangular pixels will be resampled to square pixels.
        logger.info(
            "%s/%s has rectangular pixels (%.6g m x %.6g m); resampling to %.0f m x %.0f m with nearest neighbor.",
            frequency, polarization, pixel_x, pixel_y,
            RECTANGULAR_PIXEL_OUTPUT_RESOLUTION, RECTANGULAR_PIXEL_OUTPUT_RESOLUTION,
        )

    # Count source windows so progress messages can report a meaningful completion percentage.
    total_tiles = ((height + TILE_SIZE - 1) // TILE_SIZE) * ((width + TILE_SIZE - 1) // TILE_SIZE)
    # Describe the actual processing performed in the output raster's band metadata.
    description = "Intensity Multilook RTC" if process_gslc else "RTC Gamma0 Covariance"
    try:
        # Create the native-grid staging GeoTIFF.
        with rasterio.open(native_temp_path, "w", **profile) as dst:
            # Attach a human-readable label to the sole output band.
            dst.set_band_description(1, f"{product_type} {frequency} {polarization} {description} dB")
            # Iterate over fixed-size source windows to keep memory usage bounded.
            for tile_num, window in enumerate(source_windows(height, width), start=1):
                # Convert Rasterio window offsets to integers accepted by HDF5 slicing.
                row_start = int(window.row_off)
                # Last row (exclusive) of the tile.
                row_stop = row_start + int(window.height)
                # First column of the tile.
                col_start = int(window.col_off)
                # Last column (exclusive) of the tile.
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
                # Replace invalid pixels with the NoData value.
                db_tile = np.where(valid_db, db_tile, OUTPUT_NODATA)
                # Write this completed native-grid tile into the temporary GeoTIFF.
                dst.write(db_tile.astype(np.float32), 1, window=window)
                # Write an explicit validity mask as well as the NoData value so software
                # that prioritizes raster masks renders the swath edge transparently.
                dst.write_mask(np.where(valid_db, 255, 0).astype(np.uint8), window=window)

                # Log progress approximately ten times per layer and always after the final tile.
                if tile_num == total_tiles or tile_num % max(1, total_tiles // 10) == 0:
                    # Report the percentage of tiles processed.
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
            # Release the DEM file handle.
            dem_dataset.close()

    if source_has_square_pixels:
        # Square-pixel grids need no resampling, so promote the staging GeoTIFF directly.
        os.replace(native_temp_path, temp_path)
    else:
        # Resample only rectangular grids after the scientific processing steps and before
        # publishing the final GeoTIFF.  Nearest neighbour preserves source dB samples.
        target_profile = rectangular_grid_profile(profile, transform, width, height)
        # Open the native-grid temporary GeoTIFF.
        with rasterio.open(native_temp_path) as source_raster:
            # Create the square-pixel output GeoTIFF.
            with rasterio.open(temp_path, "w", **target_profile) as destination_raster:
                # Give the output band the same description as the native band.
                destination_raster.set_band_description(
                    1, f"{product_type} {frequency} {polarization} {description} dB"
                )
                # Resample with nearest neighbour so dB values are not blended.
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
        # Delete the native-grid temporary file once resampled.
        native_temp_path.unlink()

    # Build overviews on the actual exported grid and calculate its final statistics.
    with rasterio.open(temp_path, "r+") as output_raster:
        # Build averaged overview levels for fast display.
        output_raster.build_overviews(OVERVIEW_FACTORS, Resampling.average)
        # Record the overview resampling method.
        output_raster.update_tags(ns="rio_overview", resampling="average")
        output_raster.statistics(1, approx=False)  # Compute exact band statistics so GDAL caches them with the file.

    # Atomically replace the final output path only after the temporary GeoTIFF is complete.
    os.replace(temp_path, out_path)
    # Record the final product path for operational logs and troubleshooting.
    logger.info("Created processed %s GeoTIFF: %s", product_type, out_path)
    # Return the final GeoTIFF path to the caller.
    return out_path

# Batch entry point: process every product found below ROOT_DIRECTORY.
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
        # Report the missing input folder.
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        # Exit with a failure status.
        raise SystemExit(1)

    # Create the output directory now so every successful layer has a destination.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Keep binary flood rasters separate from the processed dB GeoTIFF files.
    FLOOD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Recursively select supported HDF5 or NetCDF files from the product directory.
    source_files = [
        path
        for path in ROOT_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # End successfully when there are no products to process in the input directory.
    if not source_files:
        # Warn that there is nothing to process.
        logger.warning("No supported products found.")
        # Stop without error.
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
                # Record the original file's metadata before any layer is processed.
                log_product_metadata(product, source_file, product_type, grids, logger)

                # Consider each frequency configured at the top of this module.
                for frequency in FREQUENCIES:
                    # Skip a requested frequency that is absent from this product.
                    if frequency not in grids:
                        # Warn that this frequency is missing.
                        logger.warning("%s does not contain requested frequency %s.", source_file.name, frequency)
                        # Skip to the next frequency.
                        continue

                    # Select the grid group holding this frequency's metadata and layers.
                    grid = grids[frequency]
                    # Consider each physical polarization configured at the top of this module.
                    for polarization in POLARIZATIONS:
                        # Convert the requested physical channel to its GSLC or GCOV dataset name.
                        dataset_name = dataset_name_for_polarization(product_type, polarization)
                        # Skip a requested layer that is unavailable in this product acquisition mode.
                        if dataset_name not in grid:
                            # Warn that this polarization is missing.
                            logger.warning(
                                "%s %s does not contain requested polarization %s (%s).",
                                source_file.name,
                                frequency,
                                polarization,
                                dataset_name,
                            )
                            # Skip to the next polarization.
                            continue

                        # Process and export the available layer, resampling only if its pixels are rectangular.
                        processed_path = export_layer(
                            source_file,
                            product_type,
                            frequency,
                            polarization,
                            grid,
                            run_timestamp,
                            logger,
                        )
                        # Extract flood pixels from the completed dB GeoTIFF using the
                        # fixed threshold, even when the dB layer was created by a prior run.
                        export_flood_raster(processed_path, logger)
        # Handle unreadable or incomplete HDF5 files without stopping the remaining batch.
        except OSError as exc:
            # Tell the user to download the file again.
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            # Keep the low-level error in the debug log.
            logger.debug("Truncated file details: %s", exc)
        # Log all other per-product failures with a traceback for later diagnosis.
        except Exception as exc:
            # Log the product that failed and why.
            logger.error("Failed to process product %s: %s", source_file.name, exc)
            # Keep the full traceback in the debug log.
            logger.debug("Detailed traceback:", exc_info=True)

    # Record successful completion after every discoverable input product has been attempted.
    logger.info("Export complete.")

# --- EXECUTION START ---
if __name__ == "__main__":
    # Run the batch workflow.
    main()
