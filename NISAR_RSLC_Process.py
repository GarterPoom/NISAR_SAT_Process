#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NISAR_RSLC_Process.py

Purpose
-------
Convert NISAR L1 RSLC (Range-Doppler Single Look Complex) HDF5 products into
tiled, georeferenced GeoTIFF layers expressed in decibels (dB). Every output
uses a 5 m x 5 m map grid so it lines up with the GSLC/GCOV outputs written by
NISAR_Process.py.

Why RSLC needs extra steps
--------------------------
GSLC and GCOV are already geocoded: each pixel sits on a map grid described by
xCoordinates/yCoordinates. RSLC is *not* geocoded. Its pixels are laid out in
radar geometry (rows = zero-Doppler azimuth time, columns = slant range), so
there is no map transform to copy into a GeoTIFF. The full-resolution swath
is also very large (about 53200 x 52588 complex64 samples, ~22 GB per
polarization), so it must be read in strips and reduced before geocoding.

Processing steps (per frequency, shared by all requested polarizations)
-----------------------------------------------------------------------
1. Valid-sample masking   - keep only samples inside validSamplesSubSwath*
                            and without an inputDataExceptionMask flag.
2. Intensity              - |SLC|^2. NISAR RSLC samples are calibrated so this
                            is beta0 (radar brightness).
3. Multilooking           - non-overlapping block averaging in radar geometry
                            with separate azimuth/range looks chosen so the
                            multilooked pixel is roughly square on the ground
                            (MULTILOOK_GROUND_SPACING). This is the speckle
                            reduction step and also shrinks the data ~16x.
4. Radiometric calibration- apply the calibrationInformation/geometry/beta0
                            LUT when it is not already unity.
5. Thermal noise removal  - optional: subtract the noiseEquivalentBackscatter
                            LUT (off by default).
6. Geolocation            - interpolate the metadata/geolocationGrid cube
                            (coordinateX/Y over height, azimuth time, slant
                            range) at each multilooked pixel, iterating with
                            DEM heights so terrain is placed correctly.
7. Geocoding              - warp radar geometry onto the 5 m map grid with
                            per-pixel geolocation arrays (GDAL geoloc warp),
                            one output tile at a time.
8. Terrain normalization  - local incidence angle from the DEM surface normal
                            and the radar line-of-sight vector; convert beta0
                            to gamma0 (default), sigma0, or keep beta0.
9. dB conversion + export - NoData masking, overviews, statistics, and an
                            atomic .part.tif -> .tif publish.

Requirements
------------
h5py, numpy, scipy, rasterio, pyproj (all present in the sat_process env).
"""

# --- IMPORT SECTION ---

# Import future behavior to ensure type hinting works correctly in older Python 3 versions
from __future__ import annotations

# Import logging to track script execution and errors in real-time and to files
import logging

# Import math for snapping output grid bounds to whole target-resolution pixels
import math

# Import os for operating system tasks like replacing files (os.replace)
import os

# Import shutil to remove the temporary per-product work directory
import shutil

# Import sys to interact with the interpreter (used for sys.stdout and exiting)
import sys

# Import perf_counter for precise elapsed-time measurements that are unaffected by clock changes
from time import perf_counter

# Import datetime to create unique, timestamped filenames for logs and outputs
from datetime import datetime

# Import Path from pathlib for robust, cross-platform filesystem path manipulation
from pathlib import Path

# Import h5py for reading scientific data in HDF5 format
import h5py

# Import numpy for high-performance numerical array manipulations
import numpy as np

# Import RegularGridInterpolator to interpolate the geolocation cube and calibration LUTs
from scipy.interpolate import RegularGridInterpolator

# Import map_coordinates for fast bilinear DEM sampling at arbitrary coordinates
from scipy.ndimage import map_coordinates

# Import pyproj to convert geolocation coordinates to the output map projection
from pyproj import CRS as ProjCRS
# pyproj Transformer converts coordinates between the cube CRS and the output CRS.
from pyproj import Transformer

# Import rasterio for handling geospatial raster data (reading DEMs and writing GeoTIFFs)
import rasterio

# Import Resampling for DEM alignment, geocoding, and GeoTIFF overview construction
from rasterio.enums import Resampling

# Import reproject for DEM alignment and geolocation-array geocoding
from rasterio.warp import reproject

# Import Affine to create the transformation matrix used for georeferencing pixels
from affine import Affine

# Import Window to write the geocoded output tile by tile
from rasterio.windows import Window


# --- CONFIGURATION & CONSTANTS ---

# Determine the absolute path to the directory where this script is stored
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Root directory where the RSLC granule folders are downloaded by NISAR_Download_RSLC_Track_Frame.py
ROOT_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_Product_RSLC"

# Directory where the processed, georeferenced GeoTIFF files will be saved (shared with NISAR_Process.py)
PROCESSED_DIRECTORY = SCRIPT_DIRECTORY / "GeoTIFF_Processed"

# Scratch directory for disk-backed intermediate arrays; removed after each product
WORK_DIRECTORY = PROCESSED_DIRECTORY / "_RSLC_work"

# Directory for storing execution logs for debugging and auditing
LOG_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_logs"

# Path to the Digital Elevation Model (DEM) used for geolocation and terrain normalization
LOCAL_DEM_PATH = SCRIPT_DIRECTORY / "NASA_DEM" / "NISAR_DEM_1-20260817_064201_Mosaic.tif"

# Value added to DEM heights to obtain heights above the WGS84 ellipsoid, which is the
# reference used by the RSLC geolocation cube. The NISAR DEM mosaic is already
# ellipsoidal (open sea in the Gulf of Thailand reads about -28 m), so no offset is
# needed. Set this to the local geoid undulation when using an EGM96/EGM2008 DEM.
DEM_HEIGHT_OFFSET_M = 0.0

# Internal HDF5 paths for the RSLC product
RSLC_SWATHS_PATH = "science/LSAR/RSLC/swaths"
# HDF5 group that holds the RSLC calibration, geolocation and processing metadata.
RSLC_METADATA_PATH = "science/LSAR/RSLC/metadata"

# Tuple specifying which frequency band to process (e.g., L-Band frequencyA)
FREQUENCIES = ("frequencyA",)

# Tuple specifying which polarization channels to extract (e.g., HH, HV polarization)
POLARIZATIONS = ("HH", "HV")

# A list of file extensions that the script will recognize as valid input files
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5")

# Companion HDF5 files in each granule folder that are not RSLC products
EXCLUDED_NAME_MARKERS = ("_QA_STATS",)

# Approximate number of full-resolution swath rows read per strip (rounded to whole looks)
STRIP_TARGET_ROWS = 512

# Approximate ground pixel size, in metres, of the multilooked radar-geometry image
MULTILOOK_GROUND_SPACING = 20.0

# Minimum fraction of valid samples in a look block for the multilooked pixel to be kept
MIN_VALID_LOOK_FRACTION = 0.5

# Subtract the thermal noise floor (noiseEquivalentBackscatter LUT) before normalization
APPLY_NOISE_REMOVAL = False

# Linear floor applied after noise subtraction so dark water stays valid (-50 dB)
NOISE_FLOOR_LINEAR = 1e-5

# Output backscatter convention: "gamma0" (comparable to GCOV), "sigma0", or "beta0"
RADIOMETRIC_NORMALIZATION = "gamma0"

# Local incidence angles outside this range (layover / shadow) are written as NoData
MIN_LOCAL_INCIDENCE_DEG = 1.0
# Upper limit of valid local incidence angles, in degrees (above this is shadow).
MAX_LOCAL_INCIDENCE_DEG = 89.0

# Number of DEM height refinements during geolocation (2 is enough for sub-pixel convergence)
GEOLOCATION_DEM_ITERATIONS = 2

# Output projection. None selects the WGS84 UTM zone at the scene center.
OUTPUT_EPSG: int | None = None

# Required horizontal and vertical output pixel size, in metres.
TARGET_PIXEL_SIZE = 5.0

# Output tile size (pixels) geocoded per warp call; a multiple of TILE_SIZE
GEOCODE_TILE_SIZE = 2048

# Decimation of the geolocation arrays used to find the radar region covering each tile
LOOKUP_DECIMATION = 16

# Resampling used when warping radar geometry onto the map grid
GEOCODE_RESAMPLING = Resampling.bilinear

# The dimensions (height/width) of the GeoTIFF internal blocks
TILE_SIZE = 512

# GeoTIFF value used for pixels outside the valid NISAR swath.
OUTPUT_NODATA = -9999.0

# List of overview/pyramid levels to build for the output GeoTIFFs (for fast zooming)
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

# Product identifier embedded in output filenames
PRODUCT_TYPE = "RSLC"


# --- FUNCTION DEFINITIONS ---

def format_elapsed_time(elapsed_seconds: float) -> str:
    """Convert elapsed seconds to a readable HH:MM:SS.ss duration."""
    # Treat negative elapsed time as zero.
    safe_seconds = max(0.0, float(elapsed_seconds))
    # Split into whole minutes and remaining seconds.
    total_minutes, seconds = divmod(safe_seconds, 60.0)
    # Split the minutes into hours and minutes.
    hours, minutes = divmod(int(total_minutes), 60)
    # Return the duration as HH:MM:SS.ss.
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


# Configure logging to both the terminal and a timestamped log file.
def setup_logger(log_directory: Path) -> logging.Logger:
    """Configure a logger that writes to both the console and a timestamped file."""
    # Create the log folder if it does not exist.
    log_directory.mkdir(parents=True, exist_ok=True)
    # Log file name with the run's start time.
    log_path = log_directory / f"NISAR_RSLC_Process_{datetime.now():%Y%m%d_%H%M%S}.log"

    # Named logger used by every function in this script.
    logger = logging.getLogger("NISAR_RSLC_Process")
    # Pass every message from DEBUG level upward.
    logger.setLevel(logging.DEBUG)
    # Do not also send messages to the root logger (avoids duplicate lines).
    logger.propagate = False
    # Remove handlers left from an earlier call.
    logger.handlers.clear()

    # Shared format: timestamp | level | message.
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # One handler for the terminal and one for the log file.
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        # Let each handler pass every message from DEBUG level upward.
        handler.setLevel(logging.DEBUG)
        # Apply the shared timestamp format to this handler.
        handler.setFormatter(formatter)
        # Attach the handler so messages go to the terminal and the log file.
        logger.addHandler(handler)

    # Tell the user where the log file is written.
    logger.info("Log file: %s", log_path)
    # Return the configured logger to the caller.
    return logger


# Convert an HDF5 string (bytes or numpy bytes) or any other value to text.
def decode_text(value) -> str:
    """Decode an HDF5 byte string (or numpy bytes scalar) to str."""
    # HDF5 strings are often stored as bytes.
    if isinstance(value, (bytes, np.bytes_)):
        # Decode bytes to text, replacing any invalid characters.
        return value.decode("utf-8", errors="replace")
    # Any other value is converted with str().
    return str(value)


# Return the polarizations available in a swath (e.g. ["HH", "HV"]).
def list_of_polarizations(swath: h5py.Group) -> list[str]:
    """Return the polarizations recorded for one RSLC frequency group."""
    # Older products may lack listOfPolarizations.
    if "listOfPolarizations" not in swath:
        # Fall back to the configured polarizations that exist as datasets.
        return [name for name in POLARIZATIONS if name in swath]
    # Decode each listed polarization name.
    return [decode_text(value) for value in swath["listOfPolarizations"][()]]


# Build a linear grid interpolator, flipping any descending axis so all axes ascend.
def ascending_interpolator(axes: list[np.ndarray], values: np.ndarray) -> RegularGridInterpolator:
    """Build a linear grid interpolator, flipping any axis stored in descending order."""
    # Convert every axis to float64.
    axes = [np.asarray(axis, dtype=np.float64) for axis in axes]
    # Make sure the values are a numpy array.
    values = np.asarray(values)
    # Check each axis direction.
    for dimension, axis in enumerate(axes):
        # The axis runs from high to low values.
        if axis.size > 1 and axis[0] > axis[-1]:
            # Reverse the axis so it ascends.
            axes[dimension] = axis[::-1]
            # Reverse the values along the same dimension to match.
            values = np.flip(values, axis=dimension)
    # Linear interpolation that extrapolates (fill_value=None) instead of failing outside the grid.
    return RegularGridInterpolator(axes, values, method="linear", bounds_error=False, fill_value=None)


# Limit values to the range covered by an axis.
def clamp_to_axis(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Clamp query coordinates to an interpolation axis (nearest-edge extrapolation)."""
    # Clip to the axis minimum and maximum.
    return np.clip(values, float(np.min(axis)), float(np.max(axis)))


# Interpolate a calibration/noise LUT given on (time, range) onto the multilooked pixel grid.
def interpolate_lut_2d(
    lut: np.ndarray, lut_times: np.ndarray, lut_ranges: np.ndarray,
    times: np.ndarray, ranges: np.ndarray,
) -> np.ndarray:
    """Interpolate a (time, range) LUT onto a times x ranges grid of multilooked pixels."""
    # Work in float64 for accuracy.
    lut = np.asarray(lut, dtype=np.float64)
    # Some LUTs are stored as a single range profile; broadcast them along azimuth.
    if lut.ndim == 1:
        # Interpolate the profile at each multilooked slant range.
        profile = np.interp(ranges, lut_ranges, lut)
        # Repeat the range profile for every azimuth time.
        return np.broadcast_to(profile, (times.size, ranges.size)).astype(np.float32)
    # A LUT with a single time row is also a range profile.
    if lut.shape[0] == 1 or lut_times.size == 1:
        # Interpolate its only row at each multilooked slant range.
        profile = np.interp(ranges, lut_ranges, lut[0])
        # Repeat the range profile for every azimuth time.
        return np.broadcast_to(profile, (times.size, ranges.size)).astype(np.float32)
    # Full 2-D LUT: build an interpolator over (time, range).
    interpolator = ascending_interpolator([lut_times, lut_ranges], lut)
    # Grid of (time, range) points for every multilooked pixel, clipped to the LUT extent.
    tt, rr = np.meshgrid(clamp_to_axis(times, lut_times), clamp_to_axis(ranges, lut_ranges), indexing="ij")
    # Interpolate, reshape to the grid and return as float32.
    return interpolator(np.column_stack([tt.ravel(), rr.ravel()])).reshape(tt.shape).astype(np.float32)


# Average an axis (times or ranges) over blocks of `looks` samples.
def block_mean_axis(axis: np.ndarray, looks: int) -> np.ndarray:
    """Average a 1-D coordinate axis over consecutive blocks of `looks` samples."""
    # Work in float64 for accuracy.
    axis = np.asarray(axis, dtype=np.float64)
    # Number of blocks, counting a final partial block.
    blocks = math.ceil(axis.size / looks)
    # NaN-padded buffer that holds a whole number of blocks.
    padded = np.full(blocks * looks, np.nan)
    # Copy the axis values into the buffer.
    padded[: axis.size] = axis
    # Mean of each block, ignoring the NaN padding.
    return np.nanmean(padded.reshape(blocks, looks), axis=1)


# Number of looks in azimuth and range that give about MULTILOOK_GROUND_SPACING metres.
def multilook_looks(swath: h5py.Group) -> tuple[int, int]:
    """Choose integer azimuth/range looks that give roughly square ground pixels."""
    # Native along-track pixel spacing in metres.
    along_track = float(swath["sceneCenterAlongTrackSpacing"][()])
    # Native ground-range pixel spacing in metres.
    ground_range = float(swath["sceneCenterGroundRangeSpacing"][()])
    # Azimuth looks (at least 1).
    looks_azimuth = max(1, int(round(MULTILOOK_GROUND_SPACING / along_track)))
    # Range looks (at least 1).
    looks_range = max(1, int(round(MULTILOOK_GROUND_SPACING / ground_range)))
    # Return (azimuth looks, range looks).
    return looks_azimuth, looks_range


# Build the mask of valid samples for a block of swath rows.
def valid_sample_mask(swath: h5py.Group, row_start: int, row_stop: int, width: int) -> np.ndarray:
    """Return a boolean mask of samples inside the processed sub-swaths and free of exceptions."""
    # Number of rows in this block.
    rows = row_stop - row_start
    # Column index of every sample in a row.
    columns = np.arange(width)
    # Number of sub-swaths in this frequency (1 when not stated).
    subswath_count = int(swath["numberOfSubSwaths"][()]) if "numberOfSubSwaths" in swath else 1
    # Combined mask; None until the first sub-swath is found.
    valid = None
    # Add the valid range of each sub-swath.
    for index in range(1, subswath_count + 1):
        # Dataset name of this sub-swath's valid sample limits.
        name = f"validSamplesSubSwath{index}"
        # The product has no limits for this sub-swath.
        if name not in swath:
            # Skip it.
            continue
        # Each row stores [first valid column, one past the last valid column).
        limits = swath[name][row_start:row_stop]
        # True where the column lies inside this sub-swath's valid range.
        inside = (columns[None, :] >= limits[:, 0:1]) & (columns[None, :] < limits[:, 1:2])
        # Combine with the masks of the other sub-swaths.
        valid = inside if valid is None else (valid | inside)
    # No sub-swath limits were found.
    if valid is None:
        # Treat every sample as valid.
        valid = np.ones((rows, width), dtype=bool)
    # The product flags samples with input data exceptions.
    if "inputDataExceptionMask" in swath:
        # Exclude the flagged samples.
        valid &= swath["inputDataExceptionMask"][row_start:row_stop, :] == 0
    # Return the combined validity mask.
    return valid


# Average intensity over looks_azimuth x looks_range blocks, using only valid samples.
def block_average(intensity: np.ndarray, valid: np.ndarray, looks_azimuth: int, looks_range: int) -> np.ndarray:
    """Multilook by averaging valid samples in non-overlapping looks_azimuth x looks_range blocks."""
    # Input block size.
    rows, columns = intensity.shape
    # Number of output rows (including a final partial block).
    out_rows = math.ceil(rows / looks_azimuth)
    # Number of output columns (including a final partial block).
    out_columns = math.ceil(columns / looks_range)
    # Pad to whole blocks; padded samples are marked invalid so they never contribute.
    padded_values = np.zeros((out_rows * looks_azimuth, out_columns * looks_range), dtype=np.float32)
    # Matching validity buffer, initially all invalid.
    padded_valid = np.zeros(padded_values.shape, dtype=bool)
    # Samples that are valid and finite.
    usable = valid & np.isfinite(intensity)
    # Copy usable intensity into the buffer (others contribute 0).
    padded_values[:rows, :columns] = np.where(usable, intensity, 0.0)
    # Mark which buffer samples are usable.
    padded_valid[:rows, :columns] = usable

    # Shape that groups each output pixel's look block into separate axes.
    shape = (out_rows, looks_azimuth, out_columns, looks_range)
    # Sum of usable intensity in each look block.
    summed = padded_values.reshape(shape).sum(axis=(1, 3), dtype=np.float64)
    # Number of usable samples in each look block.
    counts = padded_valid.reshape(shape).sum(axis=(1, 3))

    # Output array, NaN (NoData) by default.
    multilooked = np.full((out_rows, out_columns), np.nan, dtype=np.float32)
    # Keep blocks that have at least MIN_VALID_LOOK_FRACTION of their samples valid.
    keep = counts >= max(1, MIN_VALID_LOOK_FRACTION * looks_azimuth * looks_range)
    # Mean intensity of the usable samples.
    multilooked[keep] = (summed[keep] / counts[keep]).astype(np.float32)
    # Return the multilooked block.
    return multilooked


# Find a finished GeoTIFF from an earlier run for this layer, if one exists.
def completed_output_path(source_file: Path, frequency: str, polarization: str, logger: logging.Logger) -> Path | None:
    """Return an existing final GeoTIFF for a layer, ignoring partial outputs."""
    # Filename prefix shared by every export of this layer, regardless of run timestamp.
    output_prefix = f"{source_file.stem}_{PRODUCT_TYPE}_{frequency}_{polarization}_Processed_dB_"
    # Matching GeoTIFFs, newest first.
    candidates = sorted(
        PROCESSED_DIRECTORY.glob(f"{output_prefix}*.tif"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    # Check each candidate.
    for candidate in candidates:
        # Interrupted exports keep a .part.tif suffix.
        if candidate.name.endswith(".part.tif"):
            # Skip them.
            continue
        # Try to open the candidate.
        try:
            # Open read-only to inspect it.
            with rasterio.open(candidate) as existing_raster:
                # Accept a non-empty single-band GeoTIFF.
                if (
                    existing_raster.driver == "GTiff"
                    and existing_raster.count == 1
                    and existing_raster.width > 0
                    and existing_raster.height > 0
                    and np.isclose(abs(existing_raster.transform.a), TARGET_PIXEL_SIZE)
                    and np.isclose(abs(existing_raster.transform.e), TARGET_PIXEL_SIZE)
                ):
                    # Reuse this finished GeoTIFF.
                    return candidate
        except (OSError, rasterio.errors.RasterioError) as exc:
            # A damaged file will be regenerated.
            logger.warning("Existing GeoTIFF is unreadable and will be regenerated: %s (%s)", candidate, exc)
    # No finished GeoTIFF was found.
    return None


# --- STEP 1-3: MASKING, INTENSITY, AND MULTILOOKING ---

def multilook_swath(
    swath: h5py.Group,
    polarizations: list[str],
    looks_azimuth: int,
    looks_range: int,
    work_directory: Path,
    frequency: str,
    logger: logging.Logger,
) -> dict[str, np.memmap]:
    """Read each polarization in azimuth strips and write multilooked beta0 to disk-backed arrays."""
    # Full-resolution swath size in rows and columns.
    height, width = swath[polarizations[0]].shape
    # Number of multilooked rows.
    ml_height = math.ceil(height / looks_azimuth)
    # Number of multilooked columns.
    ml_width = math.ceil(width / looks_range)
    # Whole looks per strip keep every multilook block inside a single strip.
    strip_rows = looks_azimuth * max(1, STRIP_TARGET_ROWS // looks_azimuth)

    # One disk-backed (memory-mapped) array per polarization for the multilooked result.
    multilooked = {
        polarization: np.lib.format.open_memmap(
            work_directory / f"ml_{polarization}.npy", mode="w+", dtype=np.float32, shape=(ml_height, ml_width)
        )
        for polarization in polarizations
    }
    # Log the input size, number of looks and output size.
    logger.info(
        "%s: multilooking %d x %d samples with %d (azimuth) x %d (range) looks -> %d x %d pixels.",
        frequency, height, width, looks_azimuth, looks_range, ml_height, ml_width,
    )

    # Number of strips to process.
    total_strips = math.ceil(height / strip_rows)
    # Process the swath one strip of rows at a time.
    for strip_number, row_start in enumerate(range(0, height, strip_rows), start=1):
        # Last row (exclusive) of this strip.
        row_stop = min(height, row_start + strip_rows)
        # First multilooked row produced by this strip.
        ml_row_start = row_start // looks_azimuth
        # The mask is shared by every polarization of this frequency.
        valid = valid_sample_mask(swath, row_start, row_stop, width)
        # Process each polarization of the strip.
        for polarization in polarizations:
            # Read this strip of complex samples from the HDF5 file.
            samples = swath[polarization][row_start:row_stop, :]
            # Step 2: |SLC|^2 gives beta0 for NISAR RSLC samples.
            intensity = np.square(samples.real, dtype=np.float32) + np.square(samples.imag, dtype=np.float32)
            # Free the complex samples as soon as intensity is computed.
            del samples
            # Mark zero-filled samples invalid as well, matching NISAR_Process.py.
            block = block_average(intensity, valid & (intensity > 0), looks_azimuth, looks_range)
            # Store the multilooked rows in the disk-backed array.
            multilooked[polarization][ml_row_start: ml_row_start + block.shape[0]] = block
        # Log progress about ten times and after the last strip.
        if strip_number == total_strips or strip_number % max(1, total_strips // 10) == 0:
            # Report the percentage of strips processed.
            logger.info("%s multilook progress: %.0f%% (%d/%d strips)",
                        frequency, 100 * strip_number / total_strips, strip_number, total_strips)

    # Write every disk-backed array to disk.
    for array in multilooked.values():
        # Flush pending writes.
        array.flush()
    # Return the multilooked arrays by polarization.
    return multilooked


# --- STEP 4-5: RADIOMETRIC CALIBRATION AND THERMAL NOISE REMOVAL ---

def apply_radiometric_lut_and_noise(
    product: h5py.File,
    frequency: str,
    multilooked: dict[str, np.memmap],
    times: np.ndarray,
    ranges: np.ndarray,
    logger: logging.Logger,
) -> None:
    """Apply the beta0 calibration LUT (when not unity) and optional thermal-noise subtraction in place."""
    # HDF5 group with the calibration information.
    calibration_path = f"{RSLC_METADATA_PATH}/calibrationInformation"
    # HDF5 group with the geometry-dependent calibration LUTs.
    geometry_path = f"{calibration_path}/geometry"
    # (LUT, times, ranges) when a non-unity beta0 LUT must be applied; None otherwise.
    beta0_lut = None
    # The product contains a beta0 calibration LUT.
    if f"{geometry_path}/beta0" in product:
        # Open the geometry group.
        geometry = product[geometry_path]
        # Read the beta0 LUT values.
        lut = geometry["beta0"][()].astype(np.float64)
        # Ignore non-finite LUT values.
        finite = lut[np.isfinite(lut)]
        # Log the LUT value range.
        logger.info("beta0 calibration LUT range: %.6g to %.6g", finite.min(), finite.max())
        # A LUT of ones changes nothing.
        if np.allclose(finite, 1.0, rtol=1e-3, atol=1e-3):
            # Report that no beta0 scaling is needed.
            logger.info("beta0 LUT is unity; RSLC intensity is already beta0.")
        else:
            # Warn that the beta0 LUT is not unity and will be applied.
            logger.warning(
                "beta0 LUT is not unity; multiplying |SLC|^2 by it. Verify the result against the "
                "granule's QA report before relying on absolute values."
            )
            # Store the LUT with its time and slant range axes.
            beta0_lut = (lut, geometry["zeroDopplerTime"][()], geometry["slantRange"][()])
    else:
        # Warn that no LUT was found and |SLC|^2 is used as beta0.
        logger.warning("No calibrationInformation/geometry/beta0 LUT found; assuming |SLC|^2 is beta0.")

    # Noise LUT group; None when noise removal is off or unavailable.
    noise_group = None
    # Thermal noise removal is enabled in the settings.
    if APPLY_NOISE_REMOVAL:
        # HDF5 path of this frequency's noise-equivalent backscatter LUTs.
        noise_path = f"{calibration_path}/{frequency}/noiseEquivalentBackscatter"
        # The noise LUTs exist.
        if noise_path in product:
            # Open the noise LUT group.
            noise_group = product[noise_path]
        else:
            # Warn that noise removal is skipped.
            logger.warning("Noise removal requested but %s is missing; skipping.", noise_path)

    # Nothing to apply.
    if beta0_lut is None and noise_group is None:
        # Leave the arrays unchanged.
        return

    # Apply the corrections to each polarization.
    for polarization, array in multilooked.items():
        # (values, times, ranges) of this polarization's noise LUT, or None.
        noise_lut = None
        # Noise removal is active.
        if noise_group is not None:
            # This polarization has a noise LUT with a slant range axis.
            if polarization in noise_group and "slantRange" in noise_group:
                # Read the noise LUT values.
                values = noise_group[polarization][()].astype(np.float64)
                # Units of the noise LUT, in lower case.
                units = decode_text(noise_group[polarization].attrs.get("units", "")).lower()
                # Noise floors stored in dB are negative; convert them to linear power.
                if "db" in units or np.nanmax(values) < 0:
                    # Convert dB values to linear power.
                    values = np.power(10.0, values / 10.0)
                # Time axis of the LUT (a single time when the LUT has none).
                noise_times = noise_group["zeroDopplerTime"][()] if "zeroDopplerTime" in noise_group else np.array([0.0])
                # Store the noise LUT with its axes.
                noise_lut = (values, noise_times, noise_group["slantRange"][()])
                # Report the noise floor range being subtracted.
                logger.info("%s/%s: subtracting thermal noise floor (%.2f to %.2f dB).", frequency, polarization,
                            10 * np.log10(np.nanmin(values)), 10 * np.log10(np.nanmax(values)))
            else:
                # Warn that this polarization has no noise LUT.
                logger.warning("%s/%s: no noiseEquivalentBackscatter LUT; noise not removed.", frequency, polarization)

        # Work in row strips so the LUT grids never span the whole multilooked image.
        for row_start in range(0, array.shape[0], 1024):
            # Last row (exclusive) of this block.
            row_stop = min(array.shape[0], row_start + 1024)
            # Read the block into memory as float32.
            strip = np.asarray(array[row_start:row_stop], dtype=np.float32)
            # Azimuth times of the rows in this block.
            strip_times = times[row_start:row_stop]
            # A beta0 LUT must be applied.
            if beta0_lut is not None:
                # Scale intensity by the interpolated beta0 LUT.
                strip *= interpolate_lut_2d(beta0_lut[0], beta0_lut[1], beta0_lut[2], strip_times, ranges)
            # A noise LUT must be applied.
            if noise_lut is not None:
                # Noise floor interpolated onto this block.
                noise = interpolate_lut_2d(noise_lut[0], noise_lut[1], noise_lut[2], strip_times, ranges)
                # Only correct finite pixels.
                finite = np.isfinite(strip)
                # Subtract the noise, but never go below NOISE_FLOOR_LINEAR.
                strip[finite] = np.maximum(strip[finite] - noise[finite], NOISE_FLOOR_LINEAR)
            # Write the corrected block back to the disk-backed array.
            array[row_start:row_stop] = strip
        # Flush the corrected array to disk.
        array.flush()


# --- STEP 6: GEOLOCATION WITH THE RSLC GEOLOCATION CUBE AND DEM ---

class GeolocationCube:
    """Interpolate the RSLC geolocationGrid cube at (height, zero-Doppler time, slant range)."""

    # Read the geolocation cube axes and build interpolators for map coordinates and LOS vectors.
    def __init__(self, product: h5py.File, logger: logging.Logger):
        """Load the geolocationGrid axes and build linear interpolators over them.

        Args:
            product (h5py.File): Open NISAR RSLC product.
            logger (logging.Logger): Logger for the cube summary and time-reference warnings.
        """
        # HDF5 group of the geolocation cube.
        grid = product[f"{RSLC_METADATA_PATH}/geolocationGrid"]
        # Height axis of the cube (metres above the ellipsoid).
        self.heights = grid["heightAboveEllipsoid"][()].astype(np.float64)
        # Zero-Doppler time axis of the cube.
        self.times = grid["zeroDopplerTime"][()].astype(np.float64)
        # Slant range axis of the cube.
        self.ranges = grid["slantRange"][()].astype(np.float64)
        # EPSG code of the cube's map coordinates.
        self.epsg = int(grid["epsg"][()])
        # The three cube axes in (height, time, range) order.
        axes = [self.heights, self.times, self.ranges]
        # Interpolator for the map X coordinate.
        self._x = ascending_interpolator(axes, grid["coordinateX"][()])
        # Interpolator for the map Y coordinate.
        self._y = ascending_interpolator(axes, grid["coordinateY"][()])
        # Interpolator for the X (east) component of the line-of-sight unit vector.
        self._los_x = ascending_interpolator(axes, grid["losUnitVectorX"][()])
        # Interpolator for the Y (north) component of the line-of-sight unit vector.
        self._los_y = ascending_interpolator(axes, grid["losUnitVectorY"][()])

        # The cube and swath time axes must share a reference epoch.
        cube_units = decode_text(grid["zeroDopplerTime"].attrs.get("units", ""))
        # Time units of the swath, for comparison with the cube.
        swath_units = decode_text(product[f"{RSLC_SWATHS_PATH}/zeroDopplerTime"].attrs.get("units", ""))
        # Both units are known and they differ.
        if cube_units and swath_units and cube_units != swath_units:
            # Warn that the time references differ.
            logger.warning("Time reference differs between swath (%s) and geolocation cube (%s).",
                           swath_units, cube_units)
        # Log the cube height range and EPSG code.
        logger.info("Geolocation cube: %d heights (%.0f to %.0f m), EPSG:%d.",
                    self.heights.size, self.heights.min(), self.heights.max(), self.epsg)

    # Combine heights, times and ranges into interpolation points clipped to the cube extent.
    def _points(self, heights: np.ndarray, times: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        """Return an (N, 3) array of (height, time, range) points clipped to the cube axes.

        Args:
            heights (np.ndarray): Heights above the ellipsoid, in metres.
            times (np.ndarray): Zero-Doppler times, same shape as ``heights``.
            ranges (np.ndarray): Slant ranges in metres, same shape as ``heights``.

        Returns:
            np.ndarray: Flattened interpolation points, one row per input element.
        """
        # Stack the clipped values into an (N, 3) array.
        return np.column_stack([
            clamp_to_axis(heights.ravel(), self.heights),
            clamp_to_axis(times.ravel(), self.times),
            clamp_to_axis(ranges.ravel(), self.ranges),
        ])

    # Map coordinates (x, y) at the given heights, times and ranges.
    def coordinates(self, heights, times, ranges) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate map coordinates (in the cube's EPSG) at each (height, time, range).

        Args:
            heights (np.ndarray): Heights above the ellipsoid, in metres.
            times (np.ndarray): Zero-Doppler times, same shape as ``heights``.
            ranges (np.ndarray): Slant ranges in metres, same shape as ``heights``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Map X and Y arrays with the shape of ``heights``.
        """
        # Build the interpolation points.
        points = self._points(heights, times, ranges)
        # Interpolate x and y and reshape them to the input shape.
        return self._x(points).reshape(heights.shape), self._y(points).reshape(heights.shape)

    # Line-of-sight east and north components at the given heights, times and ranges.
    def line_of_sight(self, heights, times, ranges) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the line-of-sight unit vector's X (east) and Y (north) components.

        Args:
            heights (np.ndarray): Heights above the ellipsoid, in metres.
            times (np.ndarray): Zero-Doppler times, same shape as ``heights``.
            ranges (np.ndarray): Slant ranges in metres, same shape as ``heights``.

        Returns:
            tuple[np.ndarray, np.ndarray]: LOS X and Y components with the shape of ``heights``.
        """
        # Build the interpolation points.
        points = self._points(heights, times, ranges)
        # Interpolate both LOS components and reshape them to the input shape.
        return self._los_x(points).reshape(heights.shape), self._los_y(points).reshape(heights.shape)


# Mean reference terrain height used by the processor (0 when not stored).
def reference_terrain_height(product: h5py.File) -> float:
    """Return the mean terrain height the RSLC processor assumed (fallback where the DEM is missing)."""
    # HDF5 path of the reference terrain height.
    path = f"{RSLC_METADATA_PATH}/processingInformation/parameters/referenceTerrainHeight"
    # The product stores a reference height.
    if path in product:
        # Read the reference height values.
        values = product[path][()]
        # At least one value is finite.
        if np.any(np.isfinite(values)):
            # Return their mean.
            return float(np.nanmean(values))
    # Fall back to the ellipsoid (0 m).
    return 0.0


# Load the part of the DEM that covers this scene, in the cube CRS.
def load_scene_dem(cube: GeolocationCube, logger: logging.Logger) -> tuple[np.ndarray, Affine] | None:
    """Read the DEM covering the scene into memory, on a grid in the geolocation cube's CRS."""
    # No DEM file is available.
    if not LOCAL_DEM_PATH.exists():
        # Warn that the reference height is used instead of the DEM.
        logger.warning("DEM not found at %s. Geolocation uses the reference height and "
                       "terrain normalization uses a flat ellipsoid.", LOCAL_DEM_PATH)
        # Continue without a DEM.
        return None

    # Scene footprint from the cube corners at all heights, plus a small margin.
    h, t, r = np.meshgrid(cube.heights, cube.times[[0, -1]], cube.ranges[[0, -1]], indexing="ij")
    # Map coordinates of sample points across the scene.
    xs, ys = cube.coordinates(h, t, r)
    # CRS of the geolocation cube.
    cube_crs = ProjCRS.from_epsg(cube.epsg)
    # The cube uses latitude/longitude degrees.
    if cube_crs.is_geographic:
        # About 1 arc-second resolution with a 0.02 degree margin.
        resolution, margin = 1.0 / 3600.0, 0.02
    else:
        # 30 m resolution with a 2 km margin for projected CRSs.
        resolution, margin = 30.0, 2000.0
    # West and east edges of the DEM grid.
    left, right = xs.min() - margin, xs.max() + margin
    # South and north edges of the DEM grid.
    bottom, top = ys.min() - margin, ys.max() + margin
    # DEM grid width in pixels.
    width = int(math.ceil((right - left) / resolution))
    # DEM grid height in pixels.
    height = int(math.ceil((top - bottom) / resolution))
    # North-up transform of the DEM grid.
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

    # DEM array, NaN where no data is available.
    dem = np.full((height, width), np.nan, dtype=np.float32)
    # Open the DEM file.
    with rasterio.open(LOCAL_DEM_PATH) as dem_dataset:
        # Resample the DEM onto the scene grid in the cube CRS.
        reproject(
            source=rasterio.band(dem_dataset, 1),
            destination=dem,
            src_nodata=dem_dataset.nodata,
            dst_transform=transform,
            dst_crs=f"EPSG:{cube.epsg}",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    # Apply the configured height offset (e.g. geoid to ellipsoid).
    dem += np.float32(DEM_HEIGHT_OFFSET_M)
    # Fraction of the grid that has DEM values.
    coverage = float(np.isfinite(dem).mean())
    # Log the DEM size and coverage.
    logger.info("Loaded DEM for scene: %d x %d pixels, %.1f%% coverage.", width, height, 100 * coverage)
    # The DEM does not overlap the scene.
    if coverage == 0:
        # Warn that the reference height is used instead.
        logger.warning("DEM does not cover this scene; falling back to the reference height.")
        # Continue without a DEM.
        return None
    # Return the DEM array and its transform.
    return dem, transform


# Sample the scene DEM at map coordinates with bilinear interpolation.
def sample_dem(dem: np.ndarray, transform: Affine, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly sample the scene DEM at cube-CRS coordinates (NaN outside or on NoData)."""
    # Convert map coordinates to fractional pixel columns and rows.
    columns, rows = ~transform * (xs, ys)
    # Pixel centers sit at half-pixel offsets from the transform origin.
    return map_coordinates(dem, [rows - 0.5, columns - 0.5], order=1, mode="constant", cval=np.nan)


# Choose the output CRS: OUTPUT_EPSG if set, otherwise the UTM zone of the scene centre.
def output_crs_for_scene(cube: GeolocationCube) -> ProjCRS:
    """Choose the output projection: OUTPUT_EPSG, or the UTM zone at the scene center."""
    # An explicit output EPSG code is configured.
    if OUTPUT_EPSG is not None:
        # Use it.
        return ProjCRS.from_epsg(OUTPUT_EPSG)
    # Scene centre at 0 m height, median time and median slant range.
    center = np.array([0.0]), np.array([np.median(cube.times)]), np.array([np.median(cube.ranges)])
    # Map coordinates of the scene centre.
    x, y = cube.coordinates(*center)
    # CRS of the geolocation cube.
    cube_crs = ProjCRS.from_epsg(cube.epsg)
    # The cube is already projected.
    if not cube_crs.is_geographic:
        # Use the cube's projected CRS.
        return cube_crs
    # Scene centre longitude and latitude.
    longitude, latitude = float(x[0]), float(y[0])
    # UTM zone number from the longitude.
    zone = int((longitude + 180.0) // 6.0) + 1
    # EPSG 326xx for the northern hemisphere, 327xx for the southern.
    return ProjCRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


# Compute map coordinates and LOS vectors for every multilooked pixel (DEM-refined).
def geolocate_multilooked_grid(
    cube: GeolocationCube,
    dem: tuple[np.ndarray, Affine] | None,
    reference_height: float,
    times: np.ndarray,
    ranges: np.ndarray,
    output_crs: ProjCRS,
    work_directory: Path,
    frequency: str,
    logger: logging.Logger,
) -> dict[str, np.memmap]:
    """Compute map coordinates and line-of-sight vectors for every multilooked pixel."""
    # Multilooked grid size (rows = times, columns = ranges).
    shape = (times.size, ranges.size)
    # Disk-backed arrays for map x/y and LOS east/north.
    arrays = {
        "x": np.lib.format.open_memmap(work_directory / "geo_x.npy", mode="w+", dtype=np.float64, shape=shape),
        "y": np.lib.format.open_memmap(work_directory / "geo_y.npy", mode="w+", dtype=np.float64, shape=shape),
        "los_e": np.lib.format.open_memmap(work_directory / "los_e.npy", mode="w+", dtype=np.float32, shape=shape),
        "los_n": np.lib.format.open_memmap(work_directory / "los_n.npy", mode="w+", dtype=np.float32, shape=shape),
    }
    # Transformer from the cube CRS to the output CRS.
    to_output = Transformer.from_crs(ProjCRS.from_epsg(cube.epsg), output_crs, always_xy=True)

    # Confirm the LOS vector points from the target toward the sensor. Higher targets at a
    # fixed slant range move toward the sensor, so LOS must align with that ground shift.
    mid_time = np.array([np.median(times)])
    # Median slant range, used for the LOS direction check.
    mid_range = np.array([np.median(ranges)])
    # Ground position at 0 m height.
    low = to_output.transform(*cube.coordinates(np.array([0.0]), mid_time, mid_range))
    # Ground position at 1000 m height (moves toward the sensor).
    high = to_output.transform(*cube.coordinates(np.array([1000.0]), mid_time, mid_range))
    # LOS vector at the same point.
    los_e, los_n = cube.line_of_sight(np.array([0.0]), mid_time, mid_range)
    # Horizontal direction from target toward the sensor.
    toward_sensor = np.array([high[0][0] - low[0][0], high[1][0] - low[1][0]])
    # +1 when LOS already points toward the sensor, -1 when it must be flipped.
    los_sign = 1.0 if float(np.dot(toward_sensor, [los_e[0], los_n[0]])) >= 0 else -1.0
    # LOS points away from the sensor.
    if los_sign < 0:
        # Report that the LOS direction will be flipped.
        logger.info("Geolocation cube LOS points sensor-to-target; flipping to target-to-sensor.")

    # GDAL's geolocation warp treats each geolocation value as a pixel's top-left corner,
    # so the coordinates written for warping are evaluated half a multilooked pixel
    # earlier in time and nearer in range than the pixel center.
    time_step = float(times[1] - times[0]) if times.size > 1 else 0.0
    # Spacing between multilooked slant ranges.
    range_step = float(ranges[1] - ranges[0]) if ranges.size > 1 else 0.0
    # Slant range at the near edge of each pixel.
    corner_ranges = ranges - 0.5 * range_step

    # Rows of the multilooked grid processed per strip.
    strip_rows = 256
    # Number of strips to process.
    total_strips = math.ceil(times.size / strip_rows)
    # Process the grid one strip at a time.
    for strip_number, row_start in enumerate(range(0, times.size, strip_rows), start=1):
        # Last row (exclusive) of this strip.
        row_stop = min(times.size, row_start + strip_rows)
        # Time and range of every pixel in the strip.
        strip_times, strip_ranges = np.meshgrid(times[row_start:row_stop], ranges, indexing="ij")
        # Start from the reference height everywhere.
        heights = np.full(strip_times.shape, reference_height, dtype=np.float64)

        # Iterate: locate at the current height, look up the DEM there, and relocate.
        for _ in range(GEOLOCATION_DEM_ITERATIONS if dem is not None else 0):
            # Locate the pixels at the current heights.
            xs, ys = cube.coordinates(heights, strip_times, strip_ranges)
            # Look up the DEM height at those locations.
            dem_heights = sample_dem(dem[0], dem[1], xs, ys)
            # Use the DEM height where available, else the reference height.
            heights = np.where(np.isfinite(dem_heights), dem_heights, reference_height)

        # Time and range of each pixel's top-left corner.
        corner_times, corner_range_grid = np.meshgrid(
            times[row_start:row_stop] - 0.5 * time_step, corner_ranges, indexing="ij"
        )
        # Map coordinates of the pixel corners in the cube CRS.
        xs, ys = cube.coordinates(heights, corner_times, corner_range_grid)
        # Transform the corners to the output CRS.
        map_x, map_y = to_output.transform(xs, ys)
        # LOS vectors at the pixel centres.
        los_x, los_y = cube.line_of_sight(heights, strip_times, strip_ranges)
        # Store map x.
        arrays["x"][row_start:row_stop] = map_x
        # Store map y.
        arrays["y"][row_start:row_stop] = map_y
        # Store the LOS east component, pointing toward the sensor.
        arrays["los_e"][row_start:row_stop] = los_sign * los_x
        # Store the LOS north component, pointing toward the sensor.
        arrays["los_n"][row_start:row_stop] = los_sign * los_y

        # Log progress about ten times and after the last strip.
        if strip_number == total_strips or strip_number % max(1, total_strips // 10) == 0:
            # Report the percentage of strips processed.
            logger.info("%s geolocation progress: %.0f%% (%d/%d strips)",
                        frequency, 100 * strip_number / total_strips, strip_number, total_strips)

    # Write every disk-backed array to disk.
    for array in arrays.values():
        # Flush pending writes.
        array.flush()
    # Return the geolocation arrays.
    return arrays


# --- STEP 7-9: GEOCODING, TERRAIN NORMALIZATION, AND EXPORT ---

def radar_window_for_tile(
    bounds: tuple[float, float, float, float],
    lookup_x: np.ndarray,
    lookup_y: np.ndarray,
    margin: float,
    radar_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Find the radar-geometry row/column range whose pixels fall inside a map tile."""
    # Map bounds of the tile.
    left, bottom, right, top = bounds
    # Coarse lookup pixels that fall inside the tile plus a margin.
    inside = (
        (lookup_x >= left - margin) & (lookup_x <= right + margin)
        & (lookup_y >= bottom - margin) & (lookup_y <= top + margin)
    )
    # No radar pixel falls inside this tile.
    if not inside.any():
        # Nothing to geocode for this tile.
        return None
    # Coarse rows that contain at least one inside pixel.
    rows = np.flatnonzero(inside.any(axis=1))
    # Coarse columns that contain at least one inside pixel.
    columns = np.flatnonzero(inside.any(axis=0))
    # Decimation factor between the coarse lookup and the full grid.
    step = LOOKUP_DECIMATION
    # First full-resolution row, with one coarse step of padding.
    row_start = max(0, (rows[0] - 1) * step)
    # Last full-resolution row (exclusive), with padding.
    row_stop = min(radar_shape[0], (rows[-1] + 2) * step)
    # First full-resolution column, with padding.
    column_start = max(0, (columns[0] - 1) * step)
    # Last full-resolution column (exclusive), with padding.
    column_stop = min(radar_shape[1], (columns[-1] + 2) * step)
    # The window is too small to warp.
    if row_stop - row_start < 2 or column_stop - column_start < 2:
        # Skip it.
        return None
    # Return the radar-geometry window that feeds this tile.
    return row_start, row_stop, column_start, column_stop


# Local incidence angle between the terrain normal and the LOS vector.
def local_incidence_angle(
    los_e: np.ndarray, los_n: np.ndarray, dem_tile: np.ndarray | None, transform: Affine,
) -> np.ndarray:
    """Angle (radians) between the terrain surface normal and the target-to-sensor LOS vector."""
    # Vertical LOS component from the unit-vector length.
    los_up = np.sqrt(np.clip(1.0 - los_e ** 2 - los_n ** 2, 0.0, 1.0))
    # No DEM: assume flat terrain.
    if dem_tile is None:
        # Incidence angle on a flat surface.
        return np.arccos(np.clip(los_up, -1.0, 1.0))

    # Terrain gradients in metres per metre. Rows run along transform.e (negative = south).
    gradient_row, gradient_column = np.gradient(dem_tile.astype(np.float64))
    # East slope (dz/de) in metres per metre.
    dz_de = gradient_column / transform.a
    # North slope (dz/dn); transform.e is negative so rows map to north correctly.
    dz_dn = gradient_row / transform.e
    # Pixels where the slope is not finite.
    flat = ~np.isfinite(dz_de) | ~np.isfinite(dz_dn)
    # Treat them as flat in the east direction.
    dz_de[flat] = 0.0
    # Treat them as flat in the north direction.
    dz_dn[flat] = 0.0
    # Length of the unnormalised surface normal (-dz/de, -dz/dn, 1).
    norm = np.sqrt(dz_de ** 2 + dz_dn ** 2 + 1.0)
    # Cosine of the angle between the surface normal and the LOS vector.
    cos_local = (-dz_de * los_e - dz_dn * los_n + los_up) / norm
    # Return the angle in radians.
    return np.arccos(np.clip(cos_local, -1.0, 1.0))


# Convert beta0 to the configured convention (beta0, sigma0 or gamma0).
def normalize_backscatter(beta0: np.ndarray, local_incidence: np.ndarray) -> np.ndarray:
    """Convert beta0 to the configured backscatter convention, masking layover/shadow."""
    # Valid geometry: local incidence inside the configured limits (no layover/shadow).
    valid_geometry = (
        (local_incidence >= np.radians(MIN_LOCAL_INCIDENCE_DEG))
        & (local_incidence <= np.radians(MAX_LOCAL_INCIDENCE_DEG))
    )
    # Keep beta0 unchanged.
    if RADIOMETRIC_NORMALIZATION == "beta0":
        # Copy beta0 as float32.
        result = beta0.astype(np.float32, copy=True)
    # sigma0 = beta0 * sin(local incidence).
    elif RADIOMETRIC_NORMALIZATION == "sigma0":
        # Apply the sigma0 conversion.
        result = (beta0 * np.sin(local_incidence)).astype(np.float32)
    # gamma0 = beta0 * tan(local incidence).
    elif RADIOMETRIC_NORMALIZATION == "gamma0":
        # Apply the gamma0 conversion.
        result = (beta0 * np.tan(local_incidence)).astype(np.float32)
    else:
        # Stop on an unknown normalization setting.
        raise ValueError(f"Unsupported RADIOMETRIC_NORMALIZATION: {RADIOMETRIC_NORMALIZATION}")
    # Mask pixels with invalid geometry.
    result[~valid_geometry] = np.nan
    # Return the normalized backscatter.
    return result


# Convert linear values to decibels, leaving invalid values as NaN.
def convert_to_decibels(intensity: np.ndarray) -> np.ndarray:
    """Convert positive, finite linear values to dB; everything else becomes NaN."""
    # Output array, NaN by default.
    decibels = np.full(intensity.shape, np.nan, dtype=np.float32)
    # Only positive, finite values have a dB value.
    valid = np.isfinite(intensity) & (intensity > 0)
    # dB = 10 * log10(value).
    decibels[valid] = 10.0 * np.log10(intensity[valid])
    # Return the dB array.
    return decibels


# Geocode the multilooked beta0 to the output grid, normalize, convert to dB and write GeoTIFFs.
def geocode_and_export(
    source_file: Path,
    frequency: str,
    multilooked: dict[str, np.memmap],
    geolocation: dict[str, np.memmap],
    dem: tuple[np.ndarray, Affine] | None,
    cube_epsg: int,
    output_crs: ProjCRS,
    multilook_spacing: float,
    run_timestamp: str,
    logger: logging.Logger,
) -> list[Path]:
    """Geocode multilooked beta0 onto the 5 m grid, normalize, convert to dB, and publish GeoTIFFs."""
    # Polarizations to export.
    polarizations = list(multilooked)
    # Size of the radar-geometry grid.
    radar_shape = geolocation["x"].shape

    # Output grid: the geolocated footprint snapped outward to whole 5 m pixels.
    min_x, max_x = float(np.nanmin(geolocation["x"])), float(np.nanmax(geolocation["x"]))
    # North-south extent of the geolocated footprint.
    min_y, max_y = float(np.nanmin(geolocation["y"])), float(np.nanmax(geolocation["y"]))
    # West edge snapped down to a whole pixel.
    left = math.floor(min_x / TARGET_PIXEL_SIZE) * TARGET_PIXEL_SIZE
    # North edge snapped up to a whole pixel.
    top = math.ceil(max_y / TARGET_PIXEL_SIZE) * TARGET_PIXEL_SIZE
    # Output width in pixels.
    width = int(math.ceil((max_x - left) / TARGET_PIXEL_SIZE)) + 1
    # Output height in pixels.
    height = int(math.ceil((top - min_y) / TARGET_PIXEL_SIZE)) + 1
    # North-up transform of the output grid.
    transform = Affine(TARGET_PIXEL_SIZE, 0.0, left, 0.0, -TARGET_PIXEL_SIZE, top)
    # Output CRS in rasterio form.
    map_crs = rasterio.crs.CRS.from_wkt(output_crs.to_wkt())
    # Log the output grid size, pixel size and CRS.
    logger.info("%s output grid: %d x %d pixels at %.0f m in %s.",
                frequency, width, height, TARGET_PIXEL_SIZE, output_crs.to_string())

    # GeoTIFF creation options shared by all polarizations.
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "float32",
        "crs": map_crs, "transform": transform, "nodata": OUTPUT_NODATA, "tiled": True,
        "blockxsize": TILE_SIZE, "blockysize": TILE_SIZE, "compress": "deflate", "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    # Band description for the configured normalization.
    description = {
        "gamma0": "Intensity Multilook Geocoded Gamma0",
        "sigma0": "Intensity Multilook Geocoded Sigma0",
        "beta0": "Intensity Multilook Geocoded Beta0",
    }[RADIOMETRIC_NORMALIZATION]

    # Final output path for each polarization.
    out_paths = {
        polarization: PROCESSED_DIRECTORY / (
            f"{source_file.stem}_{PRODUCT_TYPE}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
        )
        for polarization in polarizations
    }
    # Temporary path for each polarization, used until the file is complete.
    temp_paths = {polarization: path.with_suffix(".part.tif") for polarization, path in out_paths.items()}
    # Remove temporary files left by an earlier interrupted run.
    for temp_path in temp_paths.values():
        # A leftover temporary file exists.
        if temp_path.exists():
            # Delete it.
            temp_path.unlink()

    # A coarse copy of the geolocation arrays finds the radar region feeding each tile.
    lookup_x = np.asarray(geolocation["x"][::LOOKUP_DECIMATION, ::LOOKUP_DECIMATION])
    # Coarse copy of the y coordinates.
    lookup_y = np.asarray(geolocation["y"][::LOOKUP_DECIMATION, ::LOOKUP_DECIMATION])
    # Search margin around each tile in metres.
    lookup_margin = 2.0 * LOOKUP_DECIMATION * multilook_spacing

    # Open one temporary output GeoTIFF per polarization.
    destinations = {
        polarization: rasterio.open(temp_paths[polarization], "w", **profile)
        for polarization in polarizations
    }
    # Make sure the output files are closed even if geocoding fails.
    try:
        # Name each output band.
        for polarization, destination in destinations.items():
            # Band description: product, frequency, polarization, processing and units.
            destination.set_band_description(1, f"{PRODUCT_TYPE} {frequency} {polarization} {description} dB")

        # Split the output grid into GEOCODE_TILE_SIZE tiles.
        tiles = [
            Window(column, row, min(GEOCODE_TILE_SIZE, width - column), min(GEOCODE_TILE_SIZE, height - row))
            for row in range(0, height, GEOCODE_TILE_SIZE)
            for column in range(0, width, GEOCODE_TILE_SIZE)
        ]
        # Pixels masked by the local incidence limits.
        masked_geometry_pixels = 0
        # Geocode one output tile at a time.
        for tile_number, window in enumerate(tiles, start=1):
            # Tile size in rows and columns.
            tile_shape = (int(window.height), int(window.width))
            # Transform of this tile.
            tile_transform = rasterio.windows.transform(window, transform)
            # Map bounds of this tile.
            tile_bounds = rasterio.windows.bounds(window, transform)
            # Radar-geometry window that feeds this tile, or None.
            radar_window = radar_window_for_tile(tile_bounds, lookup_x, lookup_y, lookup_margin, radar_shape)

            # Output tiles, NoData by default.
            output_tiles = {polarization: np.full(tile_shape, np.nan, dtype=np.float32) for polarization in polarizations}
            # Some radar pixels fall in this tile.
            if radar_window is not None:
                # Unpack the radar window.
                row_start, row_stop, column_start, column_stop = radar_window
                # Geolocation arrays (x, y) for the radar window.
                geoloc = np.stack([
                    geolocation["x"][row_start:row_stop, column_start:column_stop],
                    geolocation["y"][row_start:row_stop, column_start:column_stop],
                ])
                # Shared GDAL warp options for this tile.
                warp_options = dict(
                    src_geoloc_array=geoloc, src_crs=map_crs, dst_transform=tile_transform, dst_crs=map_crs,
                    dst_nodata=np.nan, resampling=GEOCODE_RESAMPLING, init_dest_nodata=True,
                )
                # Step 7: geocode beta0 for every polarization in a single warp.
                beta0_source = np.stack([
                    np.asarray(multilooked[polarization][row_start:row_stop, column_start:column_stop])
                    for polarization in polarizations
                ])
                # Destination array for the geocoded beta0 of every polarization.
                beta0_tile = np.full((len(polarizations),) + tile_shape, np.nan, dtype=np.float32)
                # Warp beta0 onto the tile.
                reproject(beta0_source, beta0_tile, src_nodata=np.nan, **warp_options)

                if np.isfinite(beta0_tile).any():
                    # The LOS field is smooth and has no NoData, so it is warped separately.
                    los_source = np.stack([
                        geolocation["los_e"][row_start:row_stop, column_start:column_stop],
                        geolocation["los_n"][row_start:row_stop, column_start:column_stop],
                    ])
                    # Destination array for the geocoded LOS components.
                    los_tile = np.full((2,) + tile_shape, np.nan, dtype=np.float32)
                    # Warp the LOS components onto the tile.
                    reproject(los_source, los_tile, **warp_options)

                    # Step 8: local incidence from the DEM normal and LOS, then normalization.
                    dem_tile = None
                    # A DEM is available.
                    if dem is not None:
                        # Destination array for the DEM tile.
                        dem_tile = np.full(tile_shape, np.nan, dtype=np.float32)
                        # Resample the DEM onto the tile.
                        reproject(
                            dem[0], dem_tile, src_transform=dem[1], src_crs=rasterio.crs.CRS.from_epsg(cube_epsg), src_nodata=np.nan,
                            dst_transform=tile_transform, dst_crs=map_crs, dst_nodata=np.nan,
                            resampling=Resampling.bilinear,
                        )
                    # Local incidence angle for every tile pixel.
                    local_incidence = local_incidence_angle(los_tile[0], los_tile[1], dem_tile, tile_transform)
                    # Normalize each polarization.
                    for index, polarization in enumerate(polarizations):
                        # Convert beta0 to the configured convention.
                        normalized = normalize_backscatter(beta0_tile[index], local_incidence)
                        # Count pixels removed by the geometry mask.
                        masked_geometry_pixels += int(np.count_nonzero(np.isfinite(beta0_tile[index]) & ~np.isfinite(normalized)))
                        # Convert to dB for output.
                        output_tiles[polarization] = convert_to_decibels(normalized)

            # Step 9: write dB values with explicit NoData and a validity mask.
            for polarization, db_tile in output_tiles.items():
                # Valid output pixels.
                valid = np.isfinite(db_tile)
                # Write dB values, with NoData where invalid.
                destinations[polarization].write(np.where(valid, db_tile, OUTPUT_NODATA).astype(np.float32), 1, window=window)
                # Write the validity mask.
                destinations[polarization].write_mask(np.where(valid, 255, 0).astype(np.uint8), window=window)

            # Log progress about ten times and after the last tile.
            if tile_number == len(tiles) or tile_number % max(1, len(tiles) // 10) == 0:
                # Report the percentage of tiles geocoded.
                logger.info("%s geocoding progress: %.0f%% (%d/%d tiles)",
                            frequency, 100 * tile_number / len(tiles), tile_number, len(tiles))
        # Some pixels were masked as layover or shadow.
        if masked_geometry_pixels:
            # Report how many.
            logger.info("%s: %d pixel(s) masked as layover/shadow by the local incidence limits.",
                        frequency, masked_geometry_pixels)
    finally:
        # Close every output file.
        for destination in destinations.values():
            # Close this file.
            destination.close()

    # Final paths of the published GeoTIFFs.
    published = []
    # Finish each output file.
    for polarization in polarizations:
        # Reopen the temporary file for update.
        with rasterio.open(temp_paths[polarization], "r+") as output_raster:
            # Build averaged overview levels for fast display.
            output_raster.build_overviews(OVERVIEW_FACTORS, Resampling.average)
            # Record the overview resampling method.
            output_raster.update_tags(ns="rio_overview", resampling="average")
            # Compute exact band statistics so GDAL caches them with the file.
            output_raster.statistics(1, approx=False)
        # Rename the finished temporary file to its final name in one step.
        os.replace(temp_paths[polarization], out_paths[polarization])
        # Report the written GeoTIFF.
        logger.info("Created processed %s GeoTIFF: %s", PRODUCT_TYPE, out_paths[polarization])
        # Remember the published path.
        published.append(out_paths[polarization])
    # Return the published GeoTIFF paths.
    return published


# Run the full RSLC chain for one frequency: multilook, calibrate, geolocate and geocode.
def process_frequency(
    product: h5py.File,
    source_file: Path,
    frequency: str,
    polarizations: list[str],
    run_timestamp: str,
    work_directory: Path,
    logger: logging.Logger,
) -> list[Path]:
    """Run the full RSLC chain for one frequency and the requested polarizations."""
    # HDF5 swath group of this frequency.
    swath = product[f"{RSLC_SWATHS_PATH}/{frequency}"]
    # Number of looks in azimuth and range.
    looks_azimuth, looks_range = multilook_looks(swath)
    # Native along-track spacing in metres.
    along_track = float(swath["sceneCenterAlongTrackSpacing"][()])
    # Native ground-range spacing in metres.
    ground_range = float(swath["sceneCenterGroundRangeSpacing"][()])
    # Approximate multilooked pixel size (the larger of both axes).
    multilook_spacing = max(along_track * looks_azimuth, ground_range * looks_range)
    # Log the native and multilooked pixel spacing.
    logger.info("%s native spacing: %.2f m along-track x %.2f m ground-range; multilooked ~%.1f m x %.1f m.",
                frequency, along_track, ground_range, along_track * looks_azimuth, ground_range * looks_range)

    # Multilooked pixel centers: the mean time/range of each look block.
    times = block_mean_axis(product[f"{RSLC_SWATHS_PATH}/zeroDopplerTime"][()], looks_azimuth)
    # Slant range of each multilooked column.
    ranges = block_mean_axis(swath["slantRange"][()], looks_range)

    # Multilook every pending polarization into disk-backed arrays.
    multilooked = multilook_swath(swath, polarizations, looks_azimuth, looks_range, work_directory, frequency, logger)
    # Apply the beta0 LUT and optional noise removal.
    apply_radiometric_lut_and_noise(product, frequency, multilooked, times, ranges, logger)

    # Load the geolocation cube.
    cube = GeolocationCube(product, logger)
    # Load the DEM covering the scene (or None).
    dem = load_scene_dem(cube, logger)
    # Reference terrain height used where no DEM value exists.
    reference_height = reference_terrain_height(product)
    # Choose the output CRS.
    output_crs = output_crs_for_scene(cube)
    # Map coordinates and LOS vectors for every multilooked pixel.
    geolocation = geolocate_multilooked_grid(
        cube, dem, reference_height, times, ranges, output_crs, work_directory, frequency, logger,
    )
    # Geocode, normalize and write the GeoTIFFs; return their paths.
    return geocode_and_export(
        source_file, frequency, multilooked, geolocation, dem, cube.epsg, output_crs,
        multilook_spacing, run_timestamp, logger,
    )


# Find every RSLC HDF5 file below the root folder, skipping companion QA files.
def find_rslc_products(root_directory: Path) -> list[Path]:
    """Return RSLC HDF5 files below the root, skipping QA companions and partial downloads."""
    # Sorted list of matching files.
    return sorted(
        path
        for path in root_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(marker in path.stem for marker in EXCLUDED_NAME_MARKERS)
    )


# Batch entry point: process every RSLC product found below ROOT_DIRECTORY.
def main() -> None:
    """Discover and process every RSLC product below ROOT_DIRECTORY."""
    # Configure console and file logging.
    logger = setup_logger(LOG_DIRECTORY)

    # The input folder does not exist.
    if not ROOT_DIRECTORY.is_dir():
        # Report the missing input folder.
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        # Exit with a failure status.
        raise SystemExit(1)

    # Create the output folder.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Find the RSLC products.
    source_files = find_rslc_products(ROOT_DIRECTORY)
    # No RSLC products were found.
    if not source_files:
        # Warn that there is nothing to process.
        logger.warning("No RSLC products found in %s.", ROOT_DIRECTORY)
        # Stop without error.
        return

    # One timestamp for every output of this run.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Report how many products will be processed.
    logger.info("Found %d RSLC product(s). Starting processing...", len(source_files))
    # Log the main processing settings.
    logger.info("Normalization: %s | multilook target: %.0f m | noise removal: %s | output: %.0f m",
                RADIOMETRIC_NORMALIZATION, MULTILOOK_GROUND_SPACING, APPLY_NOISE_REMOVAL, TARGET_PIXEL_SIZE)
    # Start time of the whole batch.
    batch_start_time = perf_counter()

    # Process each product independently.
    for source_file in source_files:
        # Start time of this product.
        file_start_time = perf_counter()
        # Announce the start of this product's timer.
        logger.info("Starting timer for file: %s", source_file.name)
        # Errors in one product must not stop the others.
        try:
            # Open the product read-only.
            with h5py.File(source_file, "r") as product:
                # The file has no RSLC swaths.
                if RSLC_SWATHS_PATH not in product:
                    # Report that the file is skipped.
                    logger.info("Skipping %s; it is not an RSLC product.", source_file.name)
                    # Move on to the next file.
                    continue
                # Report the product being processed.
                logger.info("Processing RSLC product: %s", source_file.name)
                # HDF5 group with all frequency swaths.
                swaths = product[RSLC_SWATHS_PATH]

                # Process each configured frequency.
                for frequency in FREQUENCIES:
                    # The frequency is missing from this product.
                    if frequency not in swaths:
                        # Warn that this frequency is missing.
                        logger.warning("%s does not contain requested frequency %s.", source_file.name, frequency)
                        # Skip to the next frequency.
                        continue
                    # Polarizations listed for this frequency.
                    available = list_of_polarizations(swaths[frequency])
                    # Polarizations that still need processing.
                    pending = []
                    # Check each configured polarization.
                    for polarization in POLARIZATIONS:
                        # The polarization is not available.
                        if polarization not in available or polarization not in swaths[frequency]:
                            # Warn that this polarization is missing.
                            logger.warning("%s %s does not contain requested polarization %s.",
                                           source_file.name, frequency, polarization)
                            # Skip it.
                            continue
                        # Look for a finished output from an earlier run.
                        existing_output = completed_output_path(source_file, frequency, polarization, logger)
                        # A finished output exists.
                        if existing_output is not None:
                            # Report that the layer is skipped.
                            logger.info("Skipping %s/%s; completed GeoTIFF already exists: %s",
                                        frequency, polarization, existing_output)
                            # Skip it.
                            continue
                        # Queue the polarization for processing.
                        pending.append(polarization)
                    # Nothing to process for this frequency.
                    if not pending:
                        # Skip to the next frequency.
                        continue

                    # Scratch folder for this product's disk-backed arrays.
                    work_directory = WORK_DIRECTORY / f"{source_file.stem}_{frequency}"
                    # Remove scratch data left by an earlier run.
                    shutil.rmtree(work_directory, ignore_errors=True)
                    # Create an empty scratch folder.
                    work_directory.mkdir(parents=True, exist_ok=True)
                    # Make sure the scratch folder is removed even if processing fails.
                    try:
                        # Process all pending polarizations of this frequency.
                        process_frequency(product, source_file, frequency, pending, run_timestamp,
                                          work_directory, logger)
                    finally:
                        # Delete the scratch folder.
                        shutil.rmtree(work_directory, ignore_errors=True)
        except OSError as exc:
            # Tell the user to download the file again.
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            # Keep the low-level error in the debug log.
            logger.debug("Truncated file details: %s", exc)
        except Exception as exc:
            # Log the product that failed and why.
            logger.error("Failed to process product %s: %s", source_file.name, exc)
            # Keep the full traceback in the debug log.
            logger.debug("Detailed traceback:", exc_info=True)
        finally:
            # Time spent on this product.
            file_elapsed_seconds = perf_counter() - file_start_time
            # Log the product's processing time.
            logger.info("Processing time for %s: %s (%.2f seconds)",
                        source_file.name, format_elapsed_time(file_elapsed_seconds), file_elapsed_seconds)

    # Remove the whole scratch folder.
    shutil.rmtree(WORK_DIRECTORY, ignore_errors=True)
    # Time spent on the whole batch.
    batch_elapsed_seconds = perf_counter() - batch_start_time
    # Log the total processing time.
    logger.info("Export complete. Total processing time for all %d file(s): %s (%.2f seconds)",
                len(source_files), format_elapsed_time(batch_elapsed_seconds), batch_elapsed_seconds)


# --- EXECUTION START ---
if __name__ == "__main__":
    # Run the batch workflow.
    main()
