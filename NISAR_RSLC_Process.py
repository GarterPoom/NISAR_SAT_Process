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
    safe_seconds = max(0.0, float(elapsed_seconds))
    total_minutes, seconds = divmod(safe_seconds, 60.0)
    hours, minutes = divmod(int(total_minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def setup_logger(log_directory: Path) -> logging.Logger:
    """Configure a logger that writes to both the console and a timestamped file."""
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"NISAR_RSLC_Process_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("NISAR_RSLC_Process")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("Log file: %s", log_path)
    return logger


def decode_text(value) -> str:
    """Decode an HDF5 byte string (or numpy bytes scalar) to str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def list_of_polarizations(swath: h5py.Group) -> list[str]:
    """Return the polarizations recorded for one RSLC frequency group."""
    if "listOfPolarizations" not in swath:
        return [name for name in POLARIZATIONS if name in swath]
    return [decode_text(value) for value in swath["listOfPolarizations"][()]]


def ascending_interpolator(axes: list[np.ndarray], values: np.ndarray) -> RegularGridInterpolator:
    """Build a linear grid interpolator, flipping any axis stored in descending order."""
    axes = [np.asarray(axis, dtype=np.float64) for axis in axes]
    values = np.asarray(values)
    for dimension, axis in enumerate(axes):
        if axis.size > 1 and axis[0] > axis[-1]:
            axes[dimension] = axis[::-1]
            values = np.flip(values, axis=dimension)
    return RegularGridInterpolator(axes, values, method="linear", bounds_error=False, fill_value=None)


def clamp_to_axis(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Clamp query coordinates to an interpolation axis (nearest-edge extrapolation)."""
    return np.clip(values, float(np.min(axis)), float(np.max(axis)))


def interpolate_lut_2d(
    lut: np.ndarray, lut_times: np.ndarray, lut_ranges: np.ndarray,
    times: np.ndarray, ranges: np.ndarray,
) -> np.ndarray:
    """Interpolate a (time, range) LUT onto a times x ranges grid of multilooked pixels."""
    lut = np.asarray(lut, dtype=np.float64)
    # Some LUTs are stored as a single range profile; broadcast them along azimuth.
    if lut.ndim == 1:
        profile = np.interp(ranges, lut_ranges, lut)
        return np.broadcast_to(profile, (times.size, ranges.size)).astype(np.float32)
    if lut.shape[0] == 1 or lut_times.size == 1:
        profile = np.interp(ranges, lut_ranges, lut[0])
        return np.broadcast_to(profile, (times.size, ranges.size)).astype(np.float32)
    interpolator = ascending_interpolator([lut_times, lut_ranges], lut)
    tt, rr = np.meshgrid(clamp_to_axis(times, lut_times), clamp_to_axis(ranges, lut_ranges), indexing="ij")
    return interpolator(np.column_stack([tt.ravel(), rr.ravel()])).reshape(tt.shape).astype(np.float32)


def block_mean_axis(axis: np.ndarray, looks: int) -> np.ndarray:
    """Average a 1-D coordinate axis over consecutive blocks of `looks` samples."""
    axis = np.asarray(axis, dtype=np.float64)
    blocks = math.ceil(axis.size / looks)
    padded = np.full(blocks * looks, np.nan)
    padded[: axis.size] = axis
    return np.nanmean(padded.reshape(blocks, looks), axis=1)


def multilook_looks(swath: h5py.Group) -> tuple[int, int]:
    """Choose integer azimuth/range looks that give roughly square ground pixels."""
    along_track = float(swath["sceneCenterAlongTrackSpacing"][()])
    ground_range = float(swath["sceneCenterGroundRangeSpacing"][()])
    looks_azimuth = max(1, int(round(MULTILOOK_GROUND_SPACING / along_track)))
    looks_range = max(1, int(round(MULTILOOK_GROUND_SPACING / ground_range)))
    return looks_azimuth, looks_range


def valid_sample_mask(swath: h5py.Group, row_start: int, row_stop: int, width: int) -> np.ndarray:
    """Return a boolean mask of samples inside the processed sub-swaths and free of exceptions."""
    rows = row_stop - row_start
    columns = np.arange(width)
    subswath_count = int(swath["numberOfSubSwaths"][()]) if "numberOfSubSwaths" in swath else 1
    valid = None
    for index in range(1, subswath_count + 1):
        name = f"validSamplesSubSwath{index}"
        if name not in swath:
            continue
        # Each row stores [first valid column, one past the last valid column).
        limits = swath[name][row_start:row_stop]
        inside = (columns[None, :] >= limits[:, 0:1]) & (columns[None, :] < limits[:, 1:2])
        valid = inside if valid is None else (valid | inside)
    if valid is None:
        valid = np.ones((rows, width), dtype=bool)
    if "inputDataExceptionMask" in swath:
        valid &= swath["inputDataExceptionMask"][row_start:row_stop, :] == 0
    return valid


def block_average(intensity: np.ndarray, valid: np.ndarray, looks_azimuth: int, looks_range: int) -> np.ndarray:
    """Multilook by averaging valid samples in non-overlapping looks_azimuth x looks_range blocks."""
    rows, columns = intensity.shape
    out_rows = math.ceil(rows / looks_azimuth)
    out_columns = math.ceil(columns / looks_range)
    # Pad to whole blocks; padded samples are marked invalid so they never contribute.
    padded_values = np.zeros((out_rows * looks_azimuth, out_columns * looks_range), dtype=np.float32)
    padded_valid = np.zeros(padded_values.shape, dtype=bool)
    usable = valid & np.isfinite(intensity)
    padded_values[:rows, :columns] = np.where(usable, intensity, 0.0)
    padded_valid[:rows, :columns] = usable

    shape = (out_rows, looks_azimuth, out_columns, looks_range)
    summed = padded_values.reshape(shape).sum(axis=(1, 3), dtype=np.float64)
    counts = padded_valid.reshape(shape).sum(axis=(1, 3))

    multilooked = np.full((out_rows, out_columns), np.nan, dtype=np.float32)
    keep = counts >= max(1, MIN_VALID_LOOK_FRACTION * looks_azimuth * looks_range)
    multilooked[keep] = (summed[keep] / counts[keep]).astype(np.float32)
    return multilooked


def completed_output_path(source_file: Path, frequency: str, polarization: str, logger: logging.Logger) -> Path | None:
    """Return an existing final GeoTIFF for a layer, ignoring partial outputs."""
    output_prefix = f"{source_file.stem}_{PRODUCT_TYPE}_{frequency}_{polarization}_Processed_dB_"
    candidates = sorted(
        PROCESSED_DIRECTORY.glob(f"{output_prefix}*.tif"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.name.endswith(".part.tif"):
            continue
        try:
            with rasterio.open(candidate) as existing_raster:
                if (
                    existing_raster.driver == "GTiff"
                    and existing_raster.count == 1
                    and existing_raster.width > 0
                    and existing_raster.height > 0
                    and np.isclose(abs(existing_raster.transform.a), TARGET_PIXEL_SIZE)
                    and np.isclose(abs(existing_raster.transform.e), TARGET_PIXEL_SIZE)
                ):
                    return candidate
        except (OSError, rasterio.errors.RasterioError) as exc:
            logger.warning("Existing GeoTIFF is unreadable and will be regenerated: %s (%s)", candidate, exc)
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
    height, width = swath[polarizations[0]].shape
    ml_height = math.ceil(height / looks_azimuth)
    ml_width = math.ceil(width / looks_range)
    # Whole looks per strip keep every multilook block inside a single strip.
    strip_rows = looks_azimuth * max(1, STRIP_TARGET_ROWS // looks_azimuth)

    multilooked = {
        polarization: np.lib.format.open_memmap(
            work_directory / f"ml_{polarization}.npy", mode="w+", dtype=np.float32, shape=(ml_height, ml_width)
        )
        for polarization in polarizations
    }
    logger.info(
        "%s: multilooking %d x %d samples with %d (azimuth) x %d (range) looks -> %d x %d pixels.",
        frequency, height, width, looks_azimuth, looks_range, ml_height, ml_width,
    )

    total_strips = math.ceil(height / strip_rows)
    for strip_number, row_start in enumerate(range(0, height, strip_rows), start=1):
        row_stop = min(height, row_start + strip_rows)
        ml_row_start = row_start // looks_azimuth
        # The mask is shared by every polarization of this frequency.
        valid = valid_sample_mask(swath, row_start, row_stop, width)
        for polarization in polarizations:
            samples = swath[polarization][row_start:row_stop, :]
            # Step 2: |SLC|^2 gives beta0 for NISAR RSLC samples.
            intensity = np.square(samples.real, dtype=np.float32) + np.square(samples.imag, dtype=np.float32)
            del samples
            # Mark zero-filled samples invalid as well, matching NISAR_Process.py.
            block = block_average(intensity, valid & (intensity > 0), looks_azimuth, looks_range)
            multilooked[polarization][ml_row_start: ml_row_start + block.shape[0]] = block
        if strip_number == total_strips or strip_number % max(1, total_strips // 10) == 0:
            logger.info("%s multilook progress: %.0f%% (%d/%d strips)",
                        frequency, 100 * strip_number / total_strips, strip_number, total_strips)

    for array in multilooked.values():
        array.flush()
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
    calibration_path = f"{RSLC_METADATA_PATH}/calibrationInformation"
    geometry_path = f"{calibration_path}/geometry"
    beta0_lut = None
    if f"{geometry_path}/beta0" in product:
        geometry = product[geometry_path]
        lut = geometry["beta0"][()].astype(np.float64)
        finite = lut[np.isfinite(lut)]
        logger.info("beta0 calibration LUT range: %.6g to %.6g", finite.min(), finite.max())
        if np.allclose(finite, 1.0, rtol=1e-3, atol=1e-3):
            logger.info("beta0 LUT is unity; RSLC intensity is already beta0.")
        else:
            logger.warning(
                "beta0 LUT is not unity; multiplying |SLC|^2 by it. Verify the result against the "
                "granule's QA report before relying on absolute values."
            )
            beta0_lut = (lut, geometry["zeroDopplerTime"][()], geometry["slantRange"][()])
    else:
        logger.warning("No calibrationInformation/geometry/beta0 LUT found; assuming |SLC|^2 is beta0.")

    noise_group = None
    if APPLY_NOISE_REMOVAL:
        noise_path = f"{calibration_path}/{frequency}/noiseEquivalentBackscatter"
        if noise_path in product:
            noise_group = product[noise_path]
        else:
            logger.warning("Noise removal requested but %s is missing; skipping.", noise_path)

    if beta0_lut is None and noise_group is None:
        return

    for polarization, array in multilooked.items():
        noise_lut = None
        if noise_group is not None:
            if polarization in noise_group and "slantRange" in noise_group:
                values = noise_group[polarization][()].astype(np.float64)
                units = decode_text(noise_group[polarization].attrs.get("units", "")).lower()
                # Noise floors stored in dB are negative; convert them to linear power.
                if "db" in units or np.nanmax(values) < 0:
                    values = np.power(10.0, values / 10.0)
                noise_times = noise_group["zeroDopplerTime"][()] if "zeroDopplerTime" in noise_group else np.array([0.0])
                noise_lut = (values, noise_times, noise_group["slantRange"][()])
                logger.info("%s/%s: subtracting thermal noise floor (%.2f to %.2f dB).", frequency, polarization,
                            10 * np.log10(np.nanmin(values)), 10 * np.log10(np.nanmax(values)))
            else:
                logger.warning("%s/%s: no noiseEquivalentBackscatter LUT; noise not removed.", frequency, polarization)

        # Work in row strips so the LUT grids never span the whole multilooked image.
        for row_start in range(0, array.shape[0], 1024):
            row_stop = min(array.shape[0], row_start + 1024)
            strip = np.asarray(array[row_start:row_stop], dtype=np.float32)
            strip_times = times[row_start:row_stop]
            if beta0_lut is not None:
                strip *= interpolate_lut_2d(beta0_lut[0], beta0_lut[1], beta0_lut[2], strip_times, ranges)
            if noise_lut is not None:
                noise = interpolate_lut_2d(noise_lut[0], noise_lut[1], noise_lut[2], strip_times, ranges)
                finite = np.isfinite(strip)
                strip[finite] = np.maximum(strip[finite] - noise[finite], NOISE_FLOOR_LINEAR)
            array[row_start:row_stop] = strip
        array.flush()


# --- STEP 6: GEOLOCATION WITH THE RSLC GEOLOCATION CUBE AND DEM ---

class GeolocationCube:
    """Interpolate the RSLC geolocationGrid cube at (height, zero-Doppler time, slant range)."""

    def __init__(self, product: h5py.File, logger: logging.Logger):
        grid = product[f"{RSLC_METADATA_PATH}/geolocationGrid"]
        self.heights = grid["heightAboveEllipsoid"][()].astype(np.float64)
        self.times = grid["zeroDopplerTime"][()].astype(np.float64)
        self.ranges = grid["slantRange"][()].astype(np.float64)
        self.epsg = int(grid["epsg"][()])
        axes = [self.heights, self.times, self.ranges]
        self._x = ascending_interpolator(axes, grid["coordinateX"][()])
        self._y = ascending_interpolator(axes, grid["coordinateY"][()])
        self._los_x = ascending_interpolator(axes, grid["losUnitVectorX"][()])
        self._los_y = ascending_interpolator(axes, grid["losUnitVectorY"][()])

        # The cube and swath time axes must share a reference epoch.
        cube_units = decode_text(grid["zeroDopplerTime"].attrs.get("units", ""))
        swath_units = decode_text(product[f"{RSLC_SWATHS_PATH}/zeroDopplerTime"].attrs.get("units", ""))
        if cube_units and swath_units and cube_units != swath_units:
            logger.warning("Time reference differs between swath (%s) and geolocation cube (%s).",
                           swath_units, cube_units)
        logger.info("Geolocation cube: %d heights (%.0f to %.0f m), EPSG:%d.",
                    self.heights.size, self.heights.min(), self.heights.max(), self.epsg)

    def _points(self, heights: np.ndarray, times: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        return np.column_stack([
            clamp_to_axis(heights.ravel(), self.heights),
            clamp_to_axis(times.ravel(), self.times),
            clamp_to_axis(ranges.ravel(), self.ranges),
        ])

    def coordinates(self, heights, times, ranges) -> tuple[np.ndarray, np.ndarray]:
        points = self._points(heights, times, ranges)
        return self._x(points).reshape(heights.shape), self._y(points).reshape(heights.shape)

    def line_of_sight(self, heights, times, ranges) -> tuple[np.ndarray, np.ndarray]:
        points = self._points(heights, times, ranges)
        return self._los_x(points).reshape(heights.shape), self._los_y(points).reshape(heights.shape)


def reference_terrain_height(product: h5py.File) -> float:
    """Return the mean terrain height the RSLC processor assumed (fallback where the DEM is missing)."""
    path = f"{RSLC_METADATA_PATH}/processingInformation/parameters/referenceTerrainHeight"
    if path in product:
        values = product[path][()]
        if np.any(np.isfinite(values)):
            return float(np.nanmean(values))
    return 0.0


def load_scene_dem(cube: GeolocationCube, logger: logging.Logger) -> tuple[np.ndarray, Affine] | None:
    """Read the DEM covering the scene into memory, on a grid in the geolocation cube's CRS."""
    if not LOCAL_DEM_PATH.exists():
        logger.warning("DEM not found at %s. Geolocation uses the reference height and "
                       "terrain normalization uses a flat ellipsoid.", LOCAL_DEM_PATH)
        return None

    # Scene footprint from the cube corners at all heights, plus a small margin.
    h, t, r = np.meshgrid(cube.heights, cube.times[[0, -1]], cube.ranges[[0, -1]], indexing="ij")
    xs, ys = cube.coordinates(h, t, r)
    cube_crs = ProjCRS.from_epsg(cube.epsg)
    if cube_crs.is_geographic:
        resolution, margin = 1.0 / 3600.0, 0.02
    else:
        resolution, margin = 30.0, 2000.0
    left, right = xs.min() - margin, xs.max() + margin
    bottom, top = ys.min() - margin, ys.max() + margin
    width = int(math.ceil((right - left) / resolution))
    height = int(math.ceil((top - bottom) / resolution))
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

    dem = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(LOCAL_DEM_PATH) as dem_dataset:
        reproject(
            source=rasterio.band(dem_dataset, 1),
            destination=dem,
            src_nodata=dem_dataset.nodata,
            dst_transform=transform,
            dst_crs=f"EPSG:{cube.epsg}",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    dem += np.float32(DEM_HEIGHT_OFFSET_M)
    coverage = float(np.isfinite(dem).mean())
    logger.info("Loaded DEM for scene: %d x %d pixels, %.1f%% coverage.", width, height, 100 * coverage)
    if coverage == 0:
        logger.warning("DEM does not cover this scene; falling back to the reference height.")
        return None
    return dem, transform


def sample_dem(dem: np.ndarray, transform: Affine, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly sample the scene DEM at cube-CRS coordinates (NaN outside or on NoData)."""
    columns, rows = ~transform * (xs, ys)
    # Pixel centers sit at half-pixel offsets from the transform origin.
    return map_coordinates(dem, [rows - 0.5, columns - 0.5], order=1, mode="constant", cval=np.nan)


def output_crs_for_scene(cube: GeolocationCube) -> ProjCRS:
    """Choose the output projection: OUTPUT_EPSG, or the UTM zone at the scene center."""
    if OUTPUT_EPSG is not None:
        return ProjCRS.from_epsg(OUTPUT_EPSG)
    center = np.array([0.0]), np.array([np.median(cube.times)]), np.array([np.median(cube.ranges)])
    x, y = cube.coordinates(*center)
    cube_crs = ProjCRS.from_epsg(cube.epsg)
    if not cube_crs.is_geographic:
        return cube_crs
    longitude, latitude = float(x[0]), float(y[0])
    zone = int((longitude + 180.0) // 6.0) + 1
    return ProjCRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


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
    shape = (times.size, ranges.size)
    arrays = {
        "x": np.lib.format.open_memmap(work_directory / "geo_x.npy", mode="w+", dtype=np.float64, shape=shape),
        "y": np.lib.format.open_memmap(work_directory / "geo_y.npy", mode="w+", dtype=np.float64, shape=shape),
        "los_e": np.lib.format.open_memmap(work_directory / "los_e.npy", mode="w+", dtype=np.float32, shape=shape),
        "los_n": np.lib.format.open_memmap(work_directory / "los_n.npy", mode="w+", dtype=np.float32, shape=shape),
    }
    to_output = Transformer.from_crs(ProjCRS.from_epsg(cube.epsg), output_crs, always_xy=True)

    # Confirm the LOS vector points from the target toward the sensor. Higher targets at a
    # fixed slant range move toward the sensor, so LOS must align with that ground shift.
    mid_time = np.array([np.median(times)])
    mid_range = np.array([np.median(ranges)])
    low = to_output.transform(*cube.coordinates(np.array([0.0]), mid_time, mid_range))
    high = to_output.transform(*cube.coordinates(np.array([1000.0]), mid_time, mid_range))
    los_e, los_n = cube.line_of_sight(np.array([0.0]), mid_time, mid_range)
    toward_sensor = np.array([high[0][0] - low[0][0], high[1][0] - low[1][0]])
    los_sign = 1.0 if float(np.dot(toward_sensor, [los_e[0], los_n[0]])) >= 0 else -1.0
    if los_sign < 0:
        logger.info("Geolocation cube LOS points sensor-to-target; flipping to target-to-sensor.")

    # GDAL's geolocation warp treats each geolocation value as a pixel's top-left corner,
    # so the coordinates written for warping are evaluated half a multilooked pixel
    # earlier in time and nearer in range than the pixel center.
    time_step = float(times[1] - times[0]) if times.size > 1 else 0.0
    range_step = float(ranges[1] - ranges[0]) if ranges.size > 1 else 0.0
    corner_ranges = ranges - 0.5 * range_step

    strip_rows = 256
    total_strips = math.ceil(times.size / strip_rows)
    for strip_number, row_start in enumerate(range(0, times.size, strip_rows), start=1):
        row_stop = min(times.size, row_start + strip_rows)
        strip_times, strip_ranges = np.meshgrid(times[row_start:row_stop], ranges, indexing="ij")
        heights = np.full(strip_times.shape, reference_height, dtype=np.float64)

        # Iterate: locate at the current height, look up the DEM there, and relocate.
        for _ in range(GEOLOCATION_DEM_ITERATIONS if dem is not None else 0):
            xs, ys = cube.coordinates(heights, strip_times, strip_ranges)
            dem_heights = sample_dem(dem[0], dem[1], xs, ys)
            heights = np.where(np.isfinite(dem_heights), dem_heights, reference_height)

        corner_times, corner_range_grid = np.meshgrid(
            times[row_start:row_stop] - 0.5 * time_step, corner_ranges, indexing="ij"
        )
        xs, ys = cube.coordinates(heights, corner_times, corner_range_grid)
        map_x, map_y = to_output.transform(xs, ys)
        los_x, los_y = cube.line_of_sight(heights, strip_times, strip_ranges)
        arrays["x"][row_start:row_stop] = map_x
        arrays["y"][row_start:row_stop] = map_y
        arrays["los_e"][row_start:row_stop] = los_sign * los_x
        arrays["los_n"][row_start:row_stop] = los_sign * los_y

        if strip_number == total_strips or strip_number % max(1, total_strips // 10) == 0:
            logger.info("%s geolocation progress: %.0f%% (%d/%d strips)",
                        frequency, 100 * strip_number / total_strips, strip_number, total_strips)

    for array in arrays.values():
        array.flush()
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
    left, bottom, right, top = bounds
    inside = (
        (lookup_x >= left - margin) & (lookup_x <= right + margin)
        & (lookup_y >= bottom - margin) & (lookup_y <= top + margin)
    )
    if not inside.any():
        return None
    rows = np.flatnonzero(inside.any(axis=1))
    columns = np.flatnonzero(inside.any(axis=0))
    step = LOOKUP_DECIMATION
    row_start = max(0, (rows[0] - 1) * step)
    row_stop = min(radar_shape[0], (rows[-1] + 2) * step)
    column_start = max(0, (columns[0] - 1) * step)
    column_stop = min(radar_shape[1], (columns[-1] + 2) * step)
    if row_stop - row_start < 2 or column_stop - column_start < 2:
        return None
    return row_start, row_stop, column_start, column_stop


def local_incidence_angle(
    los_e: np.ndarray, los_n: np.ndarray, dem_tile: np.ndarray | None, transform: Affine,
) -> np.ndarray:
    """Angle (radians) between the terrain surface normal and the target-to-sensor LOS vector."""
    los_up = np.sqrt(np.clip(1.0 - los_e ** 2 - los_n ** 2, 0.0, 1.0))
    if dem_tile is None:
        return np.arccos(np.clip(los_up, -1.0, 1.0))

    # Terrain gradients in metres per metre. Rows run along transform.e (negative = south).
    gradient_row, gradient_column = np.gradient(dem_tile.astype(np.float64))
    dz_de = gradient_column / transform.a
    dz_dn = gradient_row / transform.e
    flat = ~np.isfinite(dz_de) | ~np.isfinite(dz_dn)
    dz_de[flat] = 0.0
    dz_dn[flat] = 0.0
    norm = np.sqrt(dz_de ** 2 + dz_dn ** 2 + 1.0)
    cos_local = (-dz_de * los_e - dz_dn * los_n + los_up) / norm
    return np.arccos(np.clip(cos_local, -1.0, 1.0))


def normalize_backscatter(beta0: np.ndarray, local_incidence: np.ndarray) -> np.ndarray:
    """Convert beta0 to the configured backscatter convention, masking layover/shadow."""
    valid_geometry = (
        (local_incidence >= np.radians(MIN_LOCAL_INCIDENCE_DEG))
        & (local_incidence <= np.radians(MAX_LOCAL_INCIDENCE_DEG))
    )
    if RADIOMETRIC_NORMALIZATION == "beta0":
        result = beta0.astype(np.float32, copy=True)
    elif RADIOMETRIC_NORMALIZATION == "sigma0":
        result = (beta0 * np.sin(local_incidence)).astype(np.float32)
    elif RADIOMETRIC_NORMALIZATION == "gamma0":
        result = (beta0 * np.tan(local_incidence)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported RADIOMETRIC_NORMALIZATION: {RADIOMETRIC_NORMALIZATION}")
    result[~valid_geometry] = np.nan
    return result


def convert_to_decibels(intensity: np.ndarray) -> np.ndarray:
    """Convert positive, finite linear values to dB; everything else becomes NaN."""
    decibels = np.full(intensity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(intensity) & (intensity > 0)
    decibels[valid] = 10.0 * np.log10(intensity[valid])
    return decibels


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
    polarizations = list(multilooked)
    radar_shape = geolocation["x"].shape

    # Output grid: the geolocated footprint snapped outward to whole 5 m pixels.
    min_x, max_x = float(np.nanmin(geolocation["x"])), float(np.nanmax(geolocation["x"]))
    min_y, max_y = float(np.nanmin(geolocation["y"])), float(np.nanmax(geolocation["y"]))
    left = math.floor(min_x / TARGET_PIXEL_SIZE) * TARGET_PIXEL_SIZE
    top = math.ceil(max_y / TARGET_PIXEL_SIZE) * TARGET_PIXEL_SIZE
    width = int(math.ceil((max_x - left) / TARGET_PIXEL_SIZE)) + 1
    height = int(math.ceil((top - min_y) / TARGET_PIXEL_SIZE)) + 1
    transform = Affine(TARGET_PIXEL_SIZE, 0.0, left, 0.0, -TARGET_PIXEL_SIZE, top)
    map_crs = rasterio.crs.CRS.from_wkt(output_crs.to_wkt())
    logger.info("%s output grid: %d x %d pixels at %.0f m in %s.",
                frequency, width, height, TARGET_PIXEL_SIZE, output_crs.to_string())

    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "float32",
        "crs": map_crs, "transform": transform, "nodata": OUTPUT_NODATA, "tiled": True,
        "blockxsize": TILE_SIZE, "blockysize": TILE_SIZE, "compress": "deflate", "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    description = {
        "gamma0": "Intensity Multilook Geocoded Gamma0",
        "sigma0": "Intensity Multilook Geocoded Sigma0",
        "beta0": "Intensity Multilook Geocoded Beta0",
    }[RADIOMETRIC_NORMALIZATION]

    out_paths = {
        polarization: PROCESSED_DIRECTORY / (
            f"{source_file.stem}_{PRODUCT_TYPE}_{frequency}_{polarization}_Processed_dB_{run_timestamp}.tif"
        )
        for polarization in polarizations
    }
    temp_paths = {polarization: path.with_suffix(".part.tif") for polarization, path in out_paths.items()}
    for temp_path in temp_paths.values():
        if temp_path.exists():
            temp_path.unlink()

    # A coarse copy of the geolocation arrays finds the radar region feeding each tile.
    lookup_x = np.asarray(geolocation["x"][::LOOKUP_DECIMATION, ::LOOKUP_DECIMATION])
    lookup_y = np.asarray(geolocation["y"][::LOOKUP_DECIMATION, ::LOOKUP_DECIMATION])
    lookup_margin = 2.0 * LOOKUP_DECIMATION * multilook_spacing

    destinations = {
        polarization: rasterio.open(temp_paths[polarization], "w", **profile)
        for polarization in polarizations
    }
    try:
        for polarization, destination in destinations.items():
            destination.set_band_description(1, f"{PRODUCT_TYPE} {frequency} {polarization} {description} dB")

        tiles = [
            Window(column, row, min(GEOCODE_TILE_SIZE, width - column), min(GEOCODE_TILE_SIZE, height - row))
            for row in range(0, height, GEOCODE_TILE_SIZE)
            for column in range(0, width, GEOCODE_TILE_SIZE)
        ]
        masked_geometry_pixels = 0
        for tile_number, window in enumerate(tiles, start=1):
            tile_shape = (int(window.height), int(window.width))
            tile_transform = rasterio.windows.transform(window, transform)
            tile_bounds = rasterio.windows.bounds(window, transform)
            radar_window = radar_window_for_tile(tile_bounds, lookup_x, lookup_y, lookup_margin, radar_shape)

            output_tiles = {polarization: np.full(tile_shape, np.nan, dtype=np.float32) for polarization in polarizations}
            if radar_window is not None:
                row_start, row_stop, column_start, column_stop = radar_window
                geoloc = np.stack([
                    geolocation["x"][row_start:row_stop, column_start:column_stop],
                    geolocation["y"][row_start:row_stop, column_start:column_stop],
                ])
                warp_options = dict(
                    src_geoloc_array=geoloc, src_crs=map_crs, dst_transform=tile_transform, dst_crs=map_crs,
                    dst_nodata=np.nan, resampling=GEOCODE_RESAMPLING, init_dest_nodata=True,
                )
                # Step 7: geocode beta0 for every polarization in a single warp.
                beta0_source = np.stack([
                    np.asarray(multilooked[polarization][row_start:row_stop, column_start:column_stop])
                    for polarization in polarizations
                ])
                beta0_tile = np.full((len(polarizations),) + tile_shape, np.nan, dtype=np.float32)
                reproject(beta0_source, beta0_tile, src_nodata=np.nan, **warp_options)

                if np.isfinite(beta0_tile).any():
                    # The LOS field is smooth and has no NoData, so it is warped separately.
                    los_source = np.stack([
                        geolocation["los_e"][row_start:row_stop, column_start:column_stop],
                        geolocation["los_n"][row_start:row_stop, column_start:column_stop],
                    ])
                    los_tile = np.full((2,) + tile_shape, np.nan, dtype=np.float32)
                    reproject(los_source, los_tile, **warp_options)

                    # Step 8: local incidence from the DEM normal and LOS, then normalization.
                    dem_tile = None
                    if dem is not None:
                        dem_tile = np.full(tile_shape, np.nan, dtype=np.float32)
                        reproject(
                            dem[0], dem_tile, src_transform=dem[1], src_crs=rasterio.crs.CRS.from_epsg(cube_epsg), src_nodata=np.nan,
                            dst_transform=tile_transform, dst_crs=map_crs, dst_nodata=np.nan,
                            resampling=Resampling.bilinear,
                        )
                    local_incidence = local_incidence_angle(los_tile[0], los_tile[1], dem_tile, tile_transform)
                    for index, polarization in enumerate(polarizations):
                        normalized = normalize_backscatter(beta0_tile[index], local_incidence)
                        masked_geometry_pixels += int(np.count_nonzero(np.isfinite(beta0_tile[index]) & ~np.isfinite(normalized)))
                        output_tiles[polarization] = convert_to_decibels(normalized)

            # Step 9: write dB values with explicit NoData and a validity mask.
            for polarization, db_tile in output_tiles.items():
                valid = np.isfinite(db_tile)
                destinations[polarization].write(np.where(valid, db_tile, OUTPUT_NODATA).astype(np.float32), 1, window=window)
                destinations[polarization].write_mask(np.where(valid, 255, 0).astype(np.uint8), window=window)

            if tile_number == len(tiles) or tile_number % max(1, len(tiles) // 10) == 0:
                logger.info("%s geocoding progress: %.0f%% (%d/%d tiles)",
                            frequency, 100 * tile_number / len(tiles), tile_number, len(tiles))
        if masked_geometry_pixels:
            logger.info("%s: %d pixel(s) masked as layover/shadow by the local incidence limits.",
                        frequency, masked_geometry_pixels)
    finally:
        for destination in destinations.values():
            destination.close()

    published = []
    for polarization in polarizations:
        with rasterio.open(temp_paths[polarization], "r+") as output_raster:
            output_raster.build_overviews(OVERVIEW_FACTORS, Resampling.average)
            output_raster.update_tags(ns="rio_overview", resampling="average")
            output_raster.statistics(1, approx=False)
        os.replace(temp_paths[polarization], out_paths[polarization])
        logger.info("Created processed %s GeoTIFF: %s", PRODUCT_TYPE, out_paths[polarization])
        published.append(out_paths[polarization])
    return published


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
    swath = product[f"{RSLC_SWATHS_PATH}/{frequency}"]
    looks_azimuth, looks_range = multilook_looks(swath)
    along_track = float(swath["sceneCenterAlongTrackSpacing"][()])
    ground_range = float(swath["sceneCenterGroundRangeSpacing"][()])
    multilook_spacing = max(along_track * looks_azimuth, ground_range * looks_range)
    logger.info("%s native spacing: %.2f m along-track x %.2f m ground-range; multilooked ~%.1f m x %.1f m.",
                frequency, along_track, ground_range, along_track * looks_azimuth, ground_range * looks_range)

    # Multilooked pixel centers: the mean time/range of each look block.
    times = block_mean_axis(product[f"{RSLC_SWATHS_PATH}/zeroDopplerTime"][()], looks_azimuth)
    ranges = block_mean_axis(swath["slantRange"][()], looks_range)

    multilooked = multilook_swath(swath, polarizations, looks_azimuth, looks_range, work_directory, frequency, logger)
    apply_radiometric_lut_and_noise(product, frequency, multilooked, times, ranges, logger)

    cube = GeolocationCube(product, logger)
    dem = load_scene_dem(cube, logger)
    reference_height = reference_terrain_height(product)
    output_crs = output_crs_for_scene(cube)
    geolocation = geolocate_multilooked_grid(
        cube, dem, reference_height, times, ranges, output_crs, work_directory, frequency, logger,
    )
    return geocode_and_export(
        source_file, frequency, multilooked, geolocation, dem, cube.epsg, output_crs,
        multilook_spacing, run_timestamp, logger,
    )


def find_rslc_products(root_directory: Path) -> list[Path]:
    """Return RSLC HDF5 files below the root, skipping QA companions and partial downloads."""
    return sorted(
        path
        for path in root_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(marker in path.stem for marker in EXCLUDED_NAME_MARKERS)
    )


def main() -> None:
    """Discover and process every RSLC product below ROOT_DIRECTORY."""
    logger = setup_logger(LOG_DIRECTORY)

    if not ROOT_DIRECTORY.is_dir():
        logger.error("Input directory does not exist: %s", ROOT_DIRECTORY)
        raise SystemExit(1)

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_files = find_rslc_products(ROOT_DIRECTORY)
    if not source_files:
        logger.warning("No RSLC products found in %s.", ROOT_DIRECTORY)
        return

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Found %d RSLC product(s). Starting processing...", len(source_files))
    logger.info("Normalization: %s | multilook target: %.0f m | noise removal: %s | output: %.0f m",
                RADIOMETRIC_NORMALIZATION, MULTILOOK_GROUND_SPACING, APPLY_NOISE_REMOVAL, TARGET_PIXEL_SIZE)
    batch_start_time = perf_counter()

    for source_file in source_files:
        file_start_time = perf_counter()
        logger.info("Starting timer for file: %s", source_file.name)
        try:
            with h5py.File(source_file, "r") as product:
                if RSLC_SWATHS_PATH not in product:
                    logger.info("Skipping %s; it is not an RSLC product.", source_file.name)
                    continue
                logger.info("Processing RSLC product: %s", source_file.name)
                swaths = product[RSLC_SWATHS_PATH]

                for frequency in FREQUENCIES:
                    if frequency not in swaths:
                        logger.warning("%s does not contain requested frequency %s.", source_file.name, frequency)
                        continue
                    available = list_of_polarizations(swaths[frequency])
                    pending = []
                    for polarization in POLARIZATIONS:
                        if polarization not in available or polarization not in swaths[frequency]:
                            logger.warning("%s %s does not contain requested polarization %s.",
                                           source_file.name, frequency, polarization)
                            continue
                        existing_output = completed_output_path(source_file, frequency, polarization, logger)
                        if existing_output is not None:
                            logger.info("Skipping %s/%s; completed GeoTIFF already exists: %s",
                                        frequency, polarization, existing_output)
                            continue
                        pending.append(polarization)
                    if not pending:
                        continue

                    work_directory = WORK_DIRECTORY / f"{source_file.stem}_{frequency}"
                    shutil.rmtree(work_directory, ignore_errors=True)
                    work_directory.mkdir(parents=True, exist_ok=True)
                    try:
                        process_frequency(product, source_file, frequency, pending, run_timestamp,
                                          work_directory, logger)
                    finally:
                        shutil.rmtree(work_directory, ignore_errors=True)
        except OSError as exc:
            logger.error("File appears incomplete or corrupted. Please re-download: %s", source_file.name)
            logger.debug("Truncated file details: %s", exc)
        except Exception as exc:
            logger.error("Failed to process product %s: %s", source_file.name, exc)
            logger.debug("Detailed traceback:", exc_info=True)
        finally:
            file_elapsed_seconds = perf_counter() - file_start_time
            logger.info("Processing time for %s: %s (%.2f seconds)",
                        source_file.name, format_elapsed_time(file_elapsed_seconds), file_elapsed_seconds)

    shutil.rmtree(WORK_DIRECTORY, ignore_errors=True)
    batch_elapsed_seconds = perf_counter() - batch_start_time
    logger.info("Export complete. Total processing time for all %d file(s): %s (%.2f seconds)",
                len(source_files), format_elapsed_time(batch_elapsed_seconds), batch_elapsed_seconds)


# --- EXECUTION START ---
if __name__ == "__main__":
    main()
