#!/usr/bin/env python3
r"""Extract and mosaic NISAR SME2 soil-moisture rasters by orbit pass.

The script performs the complete workflow requested for NISAR Level-3 SME2
products:

1. Find every NISAR SME2 HDF5 file in the input folder.
2. Extract only ``/science/LSAR/SME2/grids/soilMoisture`` and its coordinate /
   projection metadata to one GeoTIFF per selected HDF5 file.
3. Separate scenes by their NISAR ``orbitPassDirection`` metadata. Ascending
   and descending scenes are never mosaicked with one another.
4. Mosaic each orbit-pass group on its native grid, without reprojection or
   resampling. All scenes in one group must therefore have the same CRS,
   pixel size, and grid alignment. The completed native mosaic is then
   reprojected to the configured output CRS.
5. Where scenes overlap, replace the overlap pixel with the arithmetic mean
   of all valid source values at that pixel.  Thus, for two valid overlapping
   scenes, the output value is ``(scene_1 + scene_2) / 2``.
6. Exclude NaN, infinity, -9999, 0, and the HDF5 dataset's declared fill value
   from the mosaic. Areas not covered by a valid input pixel remain nodata.
7. Build an external ``.ovr`` pyramid for each output for ArcGIS.

Configuration
-------------
Before running the script, edit ``INPUT_DIR`` and ``OUTPUT_PATH`` below. Then
run::

    python NISAR_SME2_Process.py

Notes
-----
The first two ``YYYYMMDDTHHMMSS`` fields in a standard SME2 filename are
treated as the acquisition start and end times. The HDF5 coordinate arrays are
pixel-center coordinates, so half a pixel is subtracted when constructing the
upper-left corner of the GeoTIFF affine transform.
"""

# Work with filename acquisition timestamps.
from datetime import datetime
# Replace completed output files atomically and inspect environment variables.
import os
# Represent input and output paths safely on Windows, macOS, and Linux.
from pathlib import Path
# Find the two standard NISAR timestamps inside each filename.
import re
# Locate and run gdaladdo when the GDAL Python bindings are unavailable.
import shutil
# Call gdaladdo without invoking a command shell.
import subprocess
# Run independent HDF5-to-GeoTIFF extractions concurrently.
from concurrent.futures import ThreadPoolExecutor, as_completed
# Hold related scene metadata in small, explicit data objects.
from dataclasses import dataclass
# Keep type annotations concise and readable.
from typing import Any, Iterable, Sequence

# Read the requested dataset and its coordinate/projection metadata from HDF5.
import h5py
# Perform efficient numeric masking and validation.
import numpy as np
# Read and write tiled GeoTIFF datasets.
import rasterio
# Describe affine raster transforms.
from affine import Affine
# Represent coordinate reference systems understood by GDAL and PROJ.
from rasterio.crs import CRS
# Reproject the completed native-grid mosaic to the requested deliverable CRS.
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

# Use the exact NISAR SME2 baseline-soil-moisture HDF5 dataset requested.
SOIL_MOISTURE_PATH = "/science/LSAR/SME2/grids/soilMoisture"
# Obtain coordinate and projection datasets from the parent grid group.
GRID_PATH = "/science/LSAR/SME2/grids"
# Read the acquisition pass from the standard NISAR product-identification group.
ORBIT_DIRECTION_PATH = "/science/LSAR/identification/orbitPassDirection"
# Allow only the two physical pass directions that must be mosaicked separately.
ORBIT_DIRECTIONS = ("Ascending", "Descending")
# Use this sentinel for invalid source pixels and output areas not covered by a
# valid source pixel.
OUTPUT_NODATA = np.float32(-9999.0)
# Match the two adjacent acquisition timestamps in a standard NISAR filename.
TIMESTAMP_PATTERN = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")
# Convert matched timestamps with a single strict, unambiguous format.
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
# Accept both common HDF5 filename extensions, regardless of letter case.
HDF5_SUFFIXES = {".h5", ".hdf5"}


# -----------------------------------------------------------------------------
# Editable run settings -- change these values before running this script.
# -----------------------------------------------------------------------------
# Folder containing the NISAR SME2 .h5/.hdf5 input products.
INPUT_DIR = Path(r"NISAR_SME2_Product")
# Folder for the final EPSG:4326 orbit-specific mosaic GeoTIFFs.
OUTPUT_DIR = Path(r"soilMoisture")
# Give each result a self-describing base name. ``main`` appends ``_Ascending``
# or ``_Descending`` before the extension, so opposite orbit directions can
# never share a mosaic.
OUTPUT_PATH = OUTPUT_DIR / "NISAR_SME2_SoilMoisture_Mosaic.tif"
# CRS of the final deliverables. Source pixels are mosaicked in their native
# EASE-Grid first, then each completed mosaic is transformed to this CRS.
OUTPUT_CRS = "EPSG:4326"

# Optional settings. These can normally remain unchanged.
EXTRACTED_DIR: Path | None = None
SOURCE_CRS: str | None = None
RECURSIVE = True
OVERWRITE = False
# Limit extraction concurrency because each worker holds a complete HDF5 raster
# plus working arrays in memory.  Two workers is deliberately conservative for
# this large monthly batch; increase only after measuring available RAM.
MAX_EXTRACTION_WORKERS = 2


@dataclass(frozen=True)
class Scene:
    """Describe one NISAR SME2 input scene.

    Attributes
    ----------
    hdf5_path:
        Full path to the original NISAR HDF5 product.
    acquisition_start:
        First acquisition timestamp parsed from the product filename.
    acquisition_end:
        Second acquisition timestamp parsed from the product filename.
    orbit_direction:
        Canonical NISAR orbit-pass direction, either ``Ascending`` or
        ``Descending``.
    """

    # Store the original file so extraction can open it later.
    hdf5_path: Path
    # Store the start time so processing order and output metadata are stable.
    acquisition_start: datetime
    # Store the end time so the mosaic's full acquisition span can be reported.
    acquisition_end: datetime
    # Keep orbit-pass metadata with the scene to prevent cross-pass mosaics.
    orbit_direction: str


@dataclass(frozen=True)
class MosaicGrid:
    """Describe the native grid shared by all mosaic pixels."""

    # Map mosaic pixel indices to native source coordinates.
    transform: Affine
    # Store the number of mosaic columns.
    width: int
    # Store the number of mosaic rows.
    height: int
    # Keep the common source CRS with the grid.
    crs: CRS


def parse_nisar_interval(filename: str) -> tuple[datetime, datetime]:
    """Parse the standard start/end acquisition timestamps from a filename.

    Parameters
    ----------
    filename:
        Basename or full name of a standard NISAR SME2 product.

    Returns
    -------
    tuple[datetime, datetime]
        Acquisition start and end times.

    Raises
    ------
    ValueError
        If two adjacent NISAR timestamps are missing or the end precedes start.
    """

    # Search for the standard pair of underscore-delimited timestamp fields.
    match = TIMESTAMP_PATTERN.search(filename)
    # Reject ambiguous files rather than guessing a processing date.
    if match is None:
        # Include the filename in the error so the user can fix or exclude it.
        raise ValueError(f"No NISAR acquisition timestamp pair in {filename!r}.")
    # Convert the first timestamp field to a datetime.
    acquisition_start = datetime.strptime(match.group(1), TIMESTAMP_FORMAT)
    # Convert the second timestamp field to a datetime.
    acquisition_end = datetime.strptime(match.group(2), TIMESTAMP_FORMAT)
    # Protect interval-overlap logic from a malformed reversed time range.
    if acquisition_end < acquisition_start:
        # Report the malformed filename instead of silently swapping its times.
        raise ValueError(f"Acquisition end precedes start in {filename!r}.")
    # Return both times for inclusive interval-overlap selection.
    return acquisition_start, acquisition_end


def read_orbit_direction(hdf5_path: Path) -> str:
    """Read and normalize the required NISAR orbit-pass direction metadata.

    A missing or unrecognized value is a hard error: guessing could combine
    ascending and descending observations in the same output mosaic.
    """

    # Open only the identification metadata needed to separate the scenes.
    with h5py.File(hdf5_path, "r") as hdf5_file:
        # Require the standard NISAR metadata path rather than inferring a pass
        # from a filename or geographic footprint.
        if ORBIT_DIRECTION_PATH not in hdf5_file:
            raise KeyError(
                f"Missing required orbit direction metadata "
                f"{ORBIT_DIRECTION_PATH} in {hdf5_path}."
            )
        # Decode the scalar HDF5 value before validating its canonical label.
        raw_direction = decode_hdf5_value(hdf5_file[ORBIT_DIRECTION_PATH][()])
    # Normalize case and surrounding whitespace while retaining a readable label.
    normalized = str(raw_direction).strip().casefold()
    # Map the valid NISAR values to stable output-name and grouping labels.
    direction_by_value = {
        "ascending": "Ascending",
        "descending": "Descending",
    }
    # Reject unexpected metadata so there is no route to a mixed-direction mosaic.
    if normalized not in direction_by_value:
        raise ValueError(
            f"Unrecognized orbit pass direction {raw_direction!r} in {hdf5_path}; "
            "expected Ascending or Descending."
        )
    # Return the normalized direction for scene grouping and output naming.
    return direction_by_value[normalized]


def discover_scenes(
    input_dir: Path,
    recursive: bool,
) -> list[Scene]:
    """Find every SME2 HDF5 scene in the input directory.

    Files that do not contain ``_SME2_`` are ignored. SME2-looking files with
    malformed timestamp fields are reported and skipped so unrelated auxiliary
    files cannot stop a large batch unexpectedly.
    """

    # Confirm that the input location exists and is a directory.
    if not input_dir.is_dir():
        # Fail before any output directories or files are created.
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    # Select recursive or top-level-only traversal from the user flag.
    candidates: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.glob("*")
    # Collect accepted scene metadata in a list that can later be sorted.
    scenes: list[Scene] = []
    # Inspect each candidate path exactly once.
    for candidate in candidates:
        # Ignore folders and non-HDF5 files immediately.
        if not candidate.is_file() or candidate.suffix.lower() not in HDF5_SUFFIXES:
            # Continue with the next candidate without nesting the main logic.
            continue
        # Ignore other NISAR product types because their internal paths differ.
        if "_SME2_" not in candidate.name.upper():
            # Continue scanning for actual SME2 products.
            continue
        # Parse timestamps while allowing one malformed filename to be skipped.
        try:
            # Read both acquisition times from the standard filename fields.
            acquisition_start, acquisition_end = parse_nisar_interval(candidate.name)
        # Catch only filename/timestamp validation errors.
        except ValueError as error:
            # Make the skip visible without terminating the whole batch.
            print(f"WARNING: {error} Skipping this file.")
            # Continue to the next candidate file.
            continue
        # Read the authoritative pass direction before accepting the scene.
        orbit_direction = read_orbit_direction(candidate)
        # Preserve the resolved absolute source path and its parsed interval.
        scenes.append(
            Scene(
                hdf5_path=candidate.resolve(),
                acquisition_start=acquisition_start,
                acquisition_end=acquisition_end,
                orbit_direction=orbit_direction,
            )
        )
    # Sort chronologically and then by path to make results reproducible.
    scenes.sort(key=lambda scene: (scene.acquisition_start, str(scene.hdf5_path)))
    # Return every discovered scene to the extraction phase.
    return scenes


def decode_hdf5_value(value: Any) -> Any:
    """Convert common HDF5 scalar wrappers and byte strings to Python values."""

    # Convert a zero-dimensional NumPy array to its scalar value.
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # Re-run decoding because the extracted scalar can still be bytes.
        return decode_hdf5_value(value.item())
    # Decode fixed-length or variable-length HDF5 byte strings as UTF-8.
    if isinstance(value, (bytes, np.bytes_)):
        # Ignore only invalid byte sequences in optional descriptive metadata.
        return value.decode("utf-8", errors="replace").strip("\x00")
    # Return already-native numbers and strings unchanged.
    return value


def crs_from_candidate(value: Any) -> CRS | None:
    """Try to construct a Rasterio CRS from one HDF5 metadata value."""

    # Normalize NumPy scalars and HDF5 byte strings first.
    value = decode_hdf5_value(value)
    # Reject empty or missing values without asking PROJ to parse them.
    if value is None or (isinstance(value, str) and not value.strip()):
        # Signal that this candidate supplied no usable CRS.
        return None
    # Interpret integer-like values as EPSG codes.
    if isinstance(value, (int, np.integer)):
        # Ask Rasterio/PROJ to validate the numeric authority code.
        try:
            # Return a standard EPSG CRS when the code is recognized.
            return CRS.from_epsg(int(value))
        # Continue searching if the scalar was not actually a valid EPSG code.
        except Exception:
            # Return no result instead of hiding later, better metadata candidates.
            return None
    # Convert floats such as 6933.0 only when they represent an exact integer.
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        # Reuse the validated integer-handling branch.
        return crs_from_candidate(int(value))
    # Convert all remaining candidates to stripped text.
    text = str(value).strip()
    # Accept strings containing only the EPSG code digits.
    if text.isdigit():
        # Reuse the integer EPSG parser.
        return crs_from_candidate(int(text))
    # Try authority strings, WKT, PROJ strings, and other PROJ-supported forms.
    try:
        # Return the parsed CRS when PROJ recognizes the metadata.
        return CRS.from_user_input(text)
    # Treat malformed optional metadata as one failed candidate, not a fatal error.
    except Exception:
        # Let the caller inspect other metadata fields or the explicit fallback.
        return None


def read_source_crs(
    grid_group: h5py.Group,
    soil_dataset: h5py.Dataset,
    fallback: str | None,
) -> CRS:
    """Read a source CRS from NISAR projection metadata or a user fallback.

    The function checks the variable's ``grid_mapping`` reference, the standard
    ``projection`` dataset, common WKT/EPSG attributes, and each projection
    dataset's scalar value. It intentionally does not silently assume an EASE
    Grid EPSG code when metadata are absent.
    """

    # List common CRS attribute names in priority order.
    preferred_attribute_names = (
        "spatial_ref",
        "crs_wkt",
        "crs_wkt1",
        "crs_wkt2",
        "epsg_code",
        "epsg",
    )
    # Begin with the standard NISAR projection dataset name.
    projection_names = ["projection"]
    # Read a CF-style grid_mapping link when the soil variable supplies one.
    if "grid_mapping" in soil_dataset.attrs:
        # Convert the linked dataset name to ordinary text.
        linked_name = str(decode_hdf5_value(soil_dataset.attrs["grid_mapping"]))
        # Prefer that explicit link over the conventional default name.
        projection_names.insert(0, linked_name.rsplit("/", 1)[-1])
    # Remove duplicate names without changing their search order.
    projection_names = list(dict.fromkeys(projection_names))
    # Inspect each referenced projection dataset that exists in the grid group.
    for projection_name in projection_names:
        # Skip missing references so the explicit fallback can still be used.
        if projection_name not in grid_group:
            # Continue with the next possible projection dataset name.
            continue
        # Obtain the HDF5 projection dataset or object.
        projection = grid_group[projection_name]
        # Test preferred projection attributes first.
        for attribute_name in preferred_attribute_names:
            # Ignore attributes that are not present.
            if attribute_name not in projection.attrs:
                # Continue to the next common CRS attribute.
                continue
            # Attempt to parse this attribute as a CRS.
            parsed = crs_from_candidate(projection.attrs[attribute_name])
            # Return the first valid, explicit CRS definition.
            if parsed is not None:
                # Finish the search as soon as a CRS is validated.
                return parsed
        # A standard NISAR projection dataset can itself contain the EPSG code.
        if isinstance(projection, h5py.Dataset) and projection.shape == ():
            # Attempt to parse the scalar dataset value as an EPSG code or WKT.
            parsed = crs_from_candidate(projection[()])
            # Return a valid scalar CRS immediately.
            if parsed is not None:
                # Finish because product metadata are preferable to a fallback.
                return parsed
    # Inspect CRS-like attributes attached directly to the soil-moisture variable.
    for attribute_name in preferred_attribute_names:
        # Skip absent variable attributes.
        if attribute_name not in soil_dataset.attrs:
            # Continue to the next possible attribute.
            continue
        # Attempt to parse the direct variable attribute.
        parsed = crs_from_candidate(soil_dataset.attrs[attribute_name])
        # Return a valid direct CRS definition.
        if parsed is not None:
            # Finish before considering the less reliable manual fallback.
            return parsed
    # Parse the user's fallback only after exhausting embedded product metadata.
    parsed_fallback = crs_from_candidate(fallback)
    # Return the manual CRS when it is valid.
    if parsed_fallback is not None:
        # Make the fallback choice visible in the processing log.
        print(f"WARNING: using --source-crs fallback {parsed_fallback}.")
        # Supply the parsed fallback to extraction.
        return parsed_fallback
    # Refuse to guess because a wrong source CRS would misplace every output pixel.
    raise ValueError(
        "No readable CRS found in the HDF5 projection metadata. "
        "Provide --source-crs (SME2 commonly uses an EASE-Grid 2.0 CRS)."
    )


def read_coordinate_vector(grid_group: h5py.Group, name: str) -> np.ndarray:
    """Read and validate a finite, regular, one-dimensional coordinate vector."""

    # Require the standard coordinate dataset to be present.
    if name not in grid_group:
        # Identify the exact missing NISAR grid path.
        raise KeyError(f"Missing HDF5 coordinate dataset: {GRID_PATH}/{name}")
    # Convert the coordinate vector to double precision for transform calculations.
    coordinates = np.asarray(grid_group[name][...], dtype=np.float64)
    # Reject multidimensional or too-short coordinates that cannot define a grid.
    if coordinates.ndim != 1 or coordinates.size < 2:
        # Report both the dataset name and observed shape.
        raise ValueError(f"{name} must be a 1-D vector with at least 2 values; got {coordinates.shape}.")
    # Reject NaN and infinity in georeferencing coordinates.
    if not np.all(np.isfinite(coordinates)):
        # A finite affine transform cannot be built from these values.
        raise ValueError(f"{name} contains NaN or infinity.")
    # Calculate every center-to-center pixel spacing.
    differences = np.diff(coordinates)
    # Use the median to avoid letting tiny floating-point noise set the spacing.
    spacing = float(np.median(differences))
    # Reject duplicate coordinates or a zero-spacing grid.
    if spacing == 0.0:
        # Explain why an affine raster cannot represent the coordinates.
        raise ValueError(f"{name} contains zero pixel spacing.")
    # Require regular spacing within strict floating-point tolerance.
    if not np.allclose(differences, spacing, rtol=1.0e-6, atol=max(abs(spacing) * 1.0e-9, 1.0e-9)):
        # A single affine transform cannot represent an irregular coordinate vector.
        raise ValueError(f"{name} is not regularly spaced.")
    # Return the validated coordinate centers.
    return coordinates


def dataset_fill_values(dataset: h5py.Dataset) -> list[float]:
    """Collect numeric HDF5 fill-value attributes in addition to 0 and -9999."""

    # Always reject the two values explicitly required by the user.
    fill_values = [0.0, -9999.0]
    # Check the two most common HDF5/netCDF missing-data attribute names.
    for attribute_name in ("_FillValue", "missing_value"):
        # Ignore an absent optional attribute.
        if attribute_name not in dataset.attrs:
            # Continue with the next conventional fill-value attribute.
            continue
        # Flatten scalar or array attributes into individual candidates.
        candidates = np.asarray(dataset.attrs[attribute_name]).reshape(-1)
        # Inspect every declared fill value.
        for candidate in candidates:
            # Convert HDF5 scalar wrappers to a Python value.
            candidate = decode_hdf5_value(candidate)
            # Keep only numeric values that can be compared to raster cells.
            try:
                # Append the value as double precision for stable comparisons.
                fill_values.append(float(candidate))
            # Skip descriptive or malformed fill-value metadata.
            except (TypeError, ValueError):
                # Continue with any remaining candidates.
                continue
    # Remove duplicates while retaining predictable order.
    return list(dict.fromkeys(fill_values))


def make_valid_mask(values: np.ndarray, fill_values: Sequence[float]) -> np.ndarray:
    """Return True only for finite pixels unequal to every configured fill value."""

    # Start by accepting only finite numbers, thereby excluding NaN and infinity.
    valid = np.isfinite(values)
    # Exclude every explicit or metadata-derived nodata value.
    for fill_value in fill_values:
        # Use exact comparison because these values are encoded sentinels, not measurements.
        valid &= values != fill_value
    # Return the Boolean mask for scaling, summation, and counting.
    return valid


def native_tiff_profile(width: int, height: int) -> dict[str, Any]:
    """Return compression and tiling options valid for a raster of this size."""

    # Begin with options valid for every GeoTIFF size.
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": float(OUTPUT_NODATA),
        "compress": "DEFLATE",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    # GeoTIFF tile dimensions must be multiples of 16 and fit small rasters.
    if width >= 16 and height >= 16:
        # Enable tiled access so mosaic windows and ArcGIS display are efficient.
        profile["tiled"] = True
        # Choose the largest multiple-of-16 tile no larger than 512 or the raster width.
        profile["blockxsize"] = min(512, (width // 16) * 16)
        # Choose the corresponding valid tile height.
        profile["blockysize"] = min(512, (height // 16) * 16)
    # Return options ready to be supplemented with CRS and transform metadata.
    return profile


def extract_scene(
    scene: Scene,
    extracted_dir: Path,
    fallback_source_crs: str | None,
    overwrite: bool,
) -> Path:
    """Extract one SME2 soil-moisture dataset to a georeferenced GeoTIFF.

    Invalid cells are converted to ``-9999`` before mosaicking. The output is
    normalized to a north-up, west-left orientation even when the HDF5 coordinate
    vectors are stored in the opposite direction.
    """

    # Give each extracted raster the unique full NISAR product stem.
    output_path = extracted_dir / f"{scene.hdf5_path.stem}_soilMoisture.tif"                                                                                                                                                                                           
    # Preserve an existing extraction unless replacement was explicitly requested.
    if output_path.exists() and not overwrite:
        # Report reuse so a later mosaic is not mistaken for a fresh extraction.
        print(f"Reusing existing extraction: {output_path}")
        # Return the existing path for mosaic validation and processing.
        return output_path
    # Open the HDF5 product read-only.
    with h5py.File(scene.hdf5_path, "r") as hdf5_file:
        # Require the exact requested baseline soil-moisture dataset.
        if SOIL_MOISTURE_PATH not in hdf5_file:
            # Stop rather than substituting one of the other SME2 algorithm layers.
            raise KeyError(f"Missing dataset {SOIL_MOISTURE_PATH} in {scene.hdf5_path}")
        # Open the soil-moisture dataset without reading unrelated HDF5 arrays.
        soil_dataset = hdf5_file[SOIL_MOISTURE_PATH]
        # Require a two-dimensional raster band.
        if soil_dataset.ndim != 2:
            # Report the unexpected shape so a future product variant can be handled explicitly.
            raise ValueError(
                f"{SOIL_MOISTURE_PATH} must be 2-D; got {soil_dataset.shape} "
                f"in {scene.hdf5_path}."
            )
        # Open the parent grid group containing coordinates and projection metadata.
        grid_group = hdf5_file[GRID_PATH]
        # Read longitude/easting pixel-center coordinates from the NISAR grid.
        x_coordinates = read_coordinate_vector(grid_group, "xCoordinates")
        # Read latitude/northing pixel-center coordinates from the NISAR grid.
        y_coordinates = read_coordinate_vector(grid_group, "yCoordinates")
        # Ensure coordinate lengths agree with the raster dimensions.
        if soil_dataset.shape != (y_coordinates.size, x_coordinates.size):
            # Reject a mismatched transform instead of producing a shifted raster.
            raise ValueError(
                f"soilMoisture shape {soil_dataset.shape} does not match "
                f"y/x coordinate sizes {(y_coordinates.size, x_coordinates.size)}."
            )
        # Read only the requested HDF5 raster values into floating-point memory.
        raw_values = np.asarray(soil_dataset[...])
        # Build invalid-data rules before any scale or offset changes sentinel values.
        fill_values = dataset_fill_values(soil_dataset)
        # Identify finite measurements that are not 0, -9999, or declared fill values.
        valid = make_valid_mask(raw_values, fill_values)
        # Convert measurements to float32, the natural storage type for soil moisture.
        values = raw_values.astype(np.float32, copy=False)
        # Read an optional netCDF scale factor, defaulting to no scaling.
        scale_factor = float(decode_hdf5_value(soil_dataset.attrs.get("scale_factor", 1.0)))
        # Read an optional netCDF offset, defaulting to no offset.
        add_offset = float(decode_hdf5_value(soil_dataset.attrs.get("add_offset", 0.0)))
        # Apply scale and offset only to valid scientific measurements.
        values = np.where(valid, values * scale_factor + add_offset, OUTPUT_NODATA).astype(np.float32)
        # Read and validate the embedded source projection or explicit fallback.
        source_crs = read_source_crs(grid_group, soil_dataset, fallback_source_crs)
    # Normalize x so columns run from west/left to east/right.
    if x_coordinates[1] < x_coordinates[0]:
        # Reverse the x coordinate vector.
        x_coordinates = x_coordinates[::-1]
        # Reverse the corresponding raster columns.
        values = values[:, ::-1]
    # Normalize y so rows run from north/top to south/bottom.
    if y_coordinates[1] > y_coordinates[0]:
        # Reverse the y coordinate vector.
        y_coordinates = y_coordinates[::-1]
        # Reverse the corresponding raster rows.
        values = values[::-1, :]
    # Calculate the now-positive x pixel spacing.
    x_spacing = float(np.median(np.diff(x_coordinates)))
    # Calculate the now-negative y pixel spacing.
    y_spacing = float(np.median(np.diff(y_coordinates)))
    # Convert pixel-center coordinates to the upper-left outer corner transform.
    transform = Affine(
        x_spacing,
        0.0,
        float(x_coordinates[0] - x_spacing / 2.0),
        0.0,
        y_spacing,
        float(y_coordinates[0] - y_spacing / 2.0),
    )
    # Build compression and tiling options that are valid for this array size.
    profile = native_tiff_profile(values.shape[1], values.shape[0])
    # Add spatial dimensions and georeferencing to the GeoTIFF profile.
    profile.update(
        width=values.shape[1],
        height=values.shape[0],
        crs=source_crs,
        transform=transform,
    )
    # Write to a unique partial file so failed jobs do not look complete.
    partial_path = output_path.with_name(output_path.name + ".partial.tif")
    # Remove only a stale partial file belonging to this exact extraction.
    partial_path.unlink(missing_ok=True)
    # Create the single-band extracted GeoTIFF.
    with rasterio.open(partial_path, "w", **profile) as destination:
        # Write all normalized soil-moisture values to band 1.
        destination.write(values, 1)
        # Preserve compact provenance useful in ArcGIS dataset properties.
        destination.update_tags(
            source_hdf5=str(scene.hdf5_path),
            source_dataset=SOIL_MOISTURE_PATH,
            acquisition_start=scene.acquisition_start.isoformat(),
            acquisition_end=scene.acquisition_end.isoformat(),
            invalid_values="NaN, infinity, -9999, 0, and declared HDF5 fill values",
        )
        # Label the only band with the extracted NISAR variable name.
        destination.set_band_description(1, "soilMoisture")
    # Atomically replace the requested extraction only after a complete write.
    os.replace(partial_path, output_path)
    # Report the completed extraction for long multi-scene jobs.
    print(f"Extracted: {output_path}")
    # Return the extracted raster path to the mosaic stage.
    return output_path


def extract_scenes_parallel(
    scenes: Sequence[Scene],
    extracted_dir: Path,
    fallback_source_crs: str | None,
    overwrite: bool,
    max_workers: int,
) -> dict[Path, Path]:
    """Extract independent scenes with a bounded memory footprint.

    ``extract_scene`` reads one complete soil-moisture array and creates
    several temporary NumPy arrays.  Therefore the executor deliberately has
    a small, fixed worker limit: at most ``max_workers`` full scenes can be in
    memory at any instant.  The later mosaic phase stays
    serial because they already keep every source dataset open and are the
    dominant memory risk.
    """

    # Reject a nonsensical setting before creating any worker threads.
    if max_workers < 1:
        raise ValueError("MAX_EXTRACTION_WORKERS must be at least 1.")
    # Do not create surplus threads when a run has only one scene.
    worker_count = min(max_workers, len(scenes))
    # Preserve an explicit empty result for callers that happen to pass none.
    if worker_count == 0:
        return {}
    # Retain outputs by their source path; callers restore chronological order.
    extracted_by_scene: dict[Path, Path] = {}
    print(f"Extracting {len(scenes)} scene(s) with {worker_count} worker(s) (memory capped).")
    # Each future performs one independent HDF5 read and one unique GeoTIFF write.
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="nisar-extract",
    ) as executor:
        futures = {
            executor.submit(
                extract_scene,
                scene,
                extracted_dir,
                fallback_source_crs,
                overwrite,
            ): scene
            for scene in scenes
        }
        # Surface worker exceptions in the main thread with their source context.
        for future in as_completed(futures):
            scene = futures[future]
            try:
                extracted_by_scene[scene.hdf5_path] = future.result()
            except Exception as error:
                # Cancel work that has not begun; already-running writes finish
                # safely to their per-scene partial files before this exception
                # leaves the executor context.
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError(f"Extraction failed for {scene.hdf5_path}") from error
    return extracted_by_scene


def calculate_mosaic_grid(
    extracted_paths: Sequence[Path],
) -> MosaicGrid:
    """Calculate a union grid without moving or interpolating any input pixel."""

    reference_transform: Affine | None = None
    reference_crs: CRS | None = None
    source_bounds: list[tuple[float, float, float, float]] = []
    for extracted_path in extracted_paths:
        with rasterio.open(extracted_path) as source:
            if source.crs is None:
                raise ValueError(f"Extracted raster has no CRS: {extracted_path}")
            transform = source.transform
            if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
                raise ValueError(
                    "An ordinary mosaic cannot use rotated grids; no resampling was "
                    f"performed. Recreate or select an unrotated source: {extracted_path}"
                )
            if transform.a <= 0.0 or transform.e >= 0.0:
                raise ValueError(f"Extracted raster is not west-left/north-up: {extracted_path}")
            if reference_transform is None:
                reference_transform = transform
                reference_crs = source.crs
            else:
                if source.crs != reference_crs or not np.isclose(transform.a, reference_transform.a) or not np.isclose(transform.e, reference_transform.e):
                    raise ValueError(
                        "An ordinary mosaic requires identical CRS and pixel size; "
                        f"no resampling was performed for {extracted_path}."
                    )
                column_offset = (transform.c - reference_transform.c) / reference_transform.a
                row_offset = (reference_transform.f - transform.f) / abs(reference_transform.e)
                if not np.isclose(column_offset, round(column_offset)) or not np.isclose(row_offset, round(row_offset)):
                    raise ValueError(
                        "An ordinary mosaic requires pixel-aligned input grids; "
                        f"no resampling was performed for {extracted_path}."
                    )
            source_bounds.append((source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top))
    if reference_transform is None or reference_crs is None:
        raise ValueError("Cannot calculate a mosaic grid from no extracted rasters.")
    left = min(bounds[0] for bounds in source_bounds)
    bottom = min(bounds[1] for bounds in source_bounds)
    right = max(bounds[2] for bounds in source_bounds)
    top = max(bounds[3] for bounds in source_bounds)
    width = int(round((right - left) / reference_transform.a))
    height = int(round((top - bottom) / abs(reference_transform.e)))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid native-grid mosaic size: {width} x {height}.")
    return MosaicGrid(
        transform=Affine(reference_transform.a, 0.0, left, 0.0, reference_transform.e, top),
        width=width,
        height=height,
        crs=reference_crs,
    )


def mosaic_extracted_rasters(
    extracted_paths: Sequence[Path],
    output_path: Path,
    grid: MosaicGrid,
    scenes: Sequence[Scene],
    overwrite: bool,
) -> None:
    """Mosaic native source pixels with a mean in every overlap.

    Each valid source pixel contributes to the output cell at its native-grid
    location.  A cell supplied by one scene retains that value.  A cell
    supplied by two valid scenes is replaced by their pixel-wise mean, i.e.
    ``(scene_1 + scene_2) / 2``.  The contribution count is retained in a
    temporary raster so the same arithmetic-mean rule remains correct if more
    than two valid scenes happen to cover a cell.
    """

    # Require one and only one pass direction in this mosaic as a final guard
    # against a future caller accidentally mixing ascending and descending data.
    orbit_directions = {scene.orbit_direction for scene in scenes}
    if len(orbit_directions) != 1:
        raise ValueError(
            "A mosaic must contain exactly one orbit pass direction; got "
            f"{sorted(orbit_directions)}."
        )
    # Use the sole validated direction in output provenance.
    orbit_direction = orbit_directions.pop()
    # Protect an existing final mosaic unless replacement is authorized.
    if output_path.exists() and not overwrite:
        # Stop before creating a partial output or changing overviews.
        raise FileExistsError(f"Output exists; set OVERWRITE = True to replace it: {output_path}")
    # Build standard compressed floating-point GeoTIFF options.
    profile = native_tiff_profile(grid.width, grid.height)
    # Add the common native-grid metadata.
    profile.update(
        width=grid.width,
        height=grid.height,
        crs=grid.crs,
        transform=grid.transform,
    )
    # Write beside the final pathname so the atomic replacement stays on one disk.
    partial_path = output_path.with_name(output_path.name + ".partial.tif")
    # Remove only this exact stale partial mosaic from an interrupted prior run.
    partial_path.unlink(missing_ok=True)
    # Keep per-pixel contribution counts on disk so a large union grid does not
    # have to fit in memory.  This is removed before the final GeoTIFF is handed
    # off, leaving only the requested single-band soil-moisture raster.
    count_partial_path = output_path.with_name(output_path.name + ".partial.count.tif")
    count_partial_path.unlink(missing_ok=True)
    count_profile = profile.copy()
    count_profile.update(dtype="uint16", nodata=0, predictor=2)
    try:
        # ``w`` datasets are write-only in Rasterio, whereas both the output
        # mean and its count must be read and updated for every source block.
        with rasterio.open(partial_path, "w+", **profile) as destination, rasterio.open(
            count_partial_path, "w+", **count_profile
        ) as contribution_counts:
            # Initialize uncovered output cells as nodata and their contribution
            # counts as zero.  No interpolation or resampling is performed.
            for _, window in destination.block_windows(1):
                destination.write(
                    np.full((int(window.height), int(window.width)), OUTPUT_NODATA, dtype=np.float32),
                    1,
                    window=window,
                )
                contribution_counts.write(
                    np.zeros((int(window.height), int(window.width)), dtype=np.uint16),
                    1,
                    window=window,
                )
            # Process each source at its exact aligned destination location.
            for extracted_path in extracted_paths:
                with rasterio.open(extracted_path) as source:
                    # Grid compatibility was validated before this function; determine
                    # exact integer offsets without any coordinate transformation.
                    column_offset = int(round((source.transform.c - grid.transform.c) / grid.transform.a))
                    row_offset = int(round((grid.transform.f - source.transform.f) / abs(grid.transform.e)))
                    for _, source_window in source.block_windows(1):
                        destination_window = Window(
                            column_offset + int(source_window.col_off),
                            row_offset + int(source_window.row_off),
                            int(source_window.width),
                            int(source_window.height),
                        )
                        source_values = source.read(1, window=source_window, masked=False)
                        destination_values = destination.read(1, window=destination_window, masked=False)
                        counts = contribution_counts.read(1, window=destination_window, masked=False)
                        valid_source = make_valid_mask(source_values, (0.0, -9999.0))
                        # uint16 permits 65,535 valid overlapping scenes at one pixel;
                        # fail instead of silently overflowing that contribution count.
                        if np.any(valid_source & (counts == np.iinfo(np.uint16).max)):
                            raise ValueError("Too many valid scene overlaps at one output pixel.")
                        existing = valid_source & (counts > 0)
                        first_value = valid_source & ~existing
                        # Use float64 for the intermediate calculation so the two-scene
                        # average is not biased by float32 multiplication or summation.
                        destination_values[existing] = (
                            destination_values[existing].astype(np.float64) * counts[existing]
                            + source_values[existing]
                        ) / (counts[existing] + 1)
                        destination_values[first_value] = source_values[first_value]
                        counts[valid_source] += 1
                        destination.write(destination_values, 1, window=destination_window)
                        contribution_counts.write(counts, 1, window=destination_window)
            # Record the no-resampling mosaic semantics directly in the output dataset.
            destination.update_tags(
                source_dataset=SOIL_MOISTURE_PATH,
                source_scene_count=str(len(extracted_paths)),
                selected_acquisition_start=min(scene.acquisition_start for scene in scenes).isoformat(),
                selected_acquisition_end=max(scene.acquisition_end for scene in scenes).isoformat(),
                orbit_pass_direction=orbit_direction,
                overlap_method="pixel-wise arithmetic mean of all valid overlapping scene values",
                invalid_values="NaN, infinity, -9999, 0, and declared HDF5 fill values",
                uncovered_pixels="-9999 nodata",
                mosaic_method="ordinary native-grid mosaic; no reprojection or resampling",
            )
            destination.set_band_description(1, "soilMoisture")
        # Atomically replace the final mosaic after every block was written successfully.
        os.replace(partial_path, output_path)
    finally:
        # The count raster is internal working state and must never be delivered.
        count_partial_path.unlink(missing_ok=True)
    # Report the new spatial grid for verification in the console log.
    print(
        f"{orbit_direction} mosaic written: {output_path} "
        f"({grid.width} columns x {grid.height} rows, {grid.crs})"
    )


def reproject_mosaic_to_output_crs(
    source_path: Path,
    output_path: Path,
    output_crs: str | CRS,
    overwrite: bool,
) -> None:
    """Reproject a completed native-grid mosaic into the final deliverable CRS.

    Nearest-neighbour sampling preserves source soil-moisture values exactly;
    it only relocates them onto a regular geographic grid. Nodata remains -9999.
    """

    target_crs = CRS.from_user_input(output_crs)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; set OVERWRITE = True to replace it: {output_path}")
    partial_path = output_path.with_name(output_path.name + ".partial.tif")
    partial_path.unlink(missing_ok=True)
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError(f"Native mosaic has no CRS: {source_path}")
        if source.crs == target_crs:
            os.replace(source_path, output_path)
            return
        transform, width, height = calculate_default_transform(
            source.crs, target_crs, source.width, source.height, *source.bounds
        )
        profile = native_tiff_profile(width, height)
        profile.update(width=width, height=height, crs=target_crs, transform=transform)
        with rasterio.open(partial_path, "w", **profile) as destination:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(destination, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=OUTPUT_NODATA,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )
            destination.update_tags(
                **source.tags(),
                native_source_crs=str(source.crs),
                output_crs=str(target_crs),
                reprojection_method="nearest neighbor",
            )
            destination.set_band_description(1, "soilMoisture")
    os.replace(partial_path, output_path)
    print(f"Reprojected mosaic written: {output_path} ({target_crs})")

def pyramid_levels(width: int, height: int) -> list[int]:
    """Return power-of-two overview factors suitable for ArcGIS display."""

    # Begin with the conventional half-resolution pyramid.
    level = 2
    # Store accepted decimation factors in ascending order.
    levels: list[int] = []
    # Add levels until the smallest overview dimension is about 256 pixels.
    while min(width, height) / level >= 256:
        # Retain this power-of-two factor.
        levels.append(level)
        # Move to the next coarser pyramid level.
        level *= 2
    # Ensure even a small raster receives a physical external .ovr when possible.
    if not levels and min(width, height) >= 2:
        # A factor of 2 is the smallest useful overview.
        levels.append(2)
    # Return the factors for GDAL BuildOverviews or gdaladdo.
    return levels


def build_external_overviews(raster_path: Path, overwrite: bool) -> Path:
    """Build and verify an external ``<raster>.ovr`` pyramid.

    The function first uses ``osgeo.gdal`` with the source opened read-only,
    which makes the GeoTIFF driver create external overviews. If those Python
    bindings are unavailable, it runs ``gdaladdo -ro``. It verifies that the
    sidecar exists instead of accepting internal overviews as success.
    """

    # ArcGIS and GDAL use this exact sidecar naming convention.
    overview_path = Path(str(raster_path) + ".ovr")
    # Protect an existing overview unless the caller authorized replacement.
    if overview_path.exists() and not overwrite:
        # Stop because rebuilding would modify an existing deliverable.
        raise FileExistsError(
            f"Overview exists; set OVERWRITE = True to replace it: {overview_path}"
        )
    # Remove only the exact old sidecar so GDAL must create a fresh external pyramid.
    if overwrite:
        # Missing files are harmless during an overwrite run.
        overview_path.unlink(missing_ok=True)
    # Read raster dimensions without loading pixel values.
    with rasterio.open(raster_path) as source:
        # Calculate appropriate power-of-two decimation factors.
        levels = pyramid_levels(source.width, source.height)
    # Reject a one-pixel raster because no reduced-resolution pyramid can exist.
    if not levels:
        # Explain the physical limitation directly.
        raise ValueError(f"Raster is too small to build an overview: {raster_path}")
    # Prefer the GDAL Python API so the workflow remains within one Python process.
    try:
        # Import lazily so extraction can give a focused error on minimal environments.
        from osgeo import gdal  # type: ignore
    # Fall back only when the separate GDAL Python package is not installed.
    except ImportError:
        # Mark the binding as unavailable for the branch below.
        gdal = None
    # Use the Python binding when it was imported successfully.
    if gdal is not None:
        # Store external overviews as compact floating-point GeoTIFF data.
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        # Use floating-point prediction for better overview compression.
        gdal.SetConfigOption("PREDICTOR_OVERVIEW", "3")
        # Permit a BigTIFF .ovr only when the overview size requires it.
        gdal.SetConfigOption("BIGTIFF_OVERVIEW", "IF_SAFER")
        # Open read-only; this is what forces GeoTIFF overviews into a sidecar.
        dataset = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        # Fail clearly if GDAL cannot open the completed mosaic.
        if dataset is None:
            # Identify the exact path that GDAL rejected.
            raise RuntimeError(f"GDAL could not open mosaic for pyramids: {raster_path}")
        # Build average-resampled overviews for continuous soil moisture values.
        error_code = dataset.BuildOverviews("AVERAGE", levels)
        # Close the GDAL dataset so all .ovr directory entries are flushed.
        dataset = None
        # A nonzero GDAL code indicates that pyramid construction failed.
        if error_code != 0:
            # Preserve the numeric code for environment-specific diagnosis.
            raise RuntimeError(f"GDAL BuildOverviews failed with code {error_code}.")
    # Otherwise try the standard GDAL command-line program.
    else:
        # Find gdaladdo on PATH without invoking a shell.
        gdaladdo = shutil.which("gdaladdo")
        # Explain the required installation if neither GDAL interface exists.
        if gdaladdo is None:
            # The final raster remains usable, but the explicit .ovr requirement is unmet.
            raise RuntimeError(
                "Cannot create an external .ovr: install the GDAL Python bindings "
                "(osgeo) or place gdaladdo on PATH."
            )
        # Assemble each argument separately to handle paths containing spaces safely.
        command = [
            gdaladdo,
            "--config",
            "COMPRESS_OVERVIEW",
            "DEFLATE",
            "--config",
            "PREDICTOR_OVERVIEW",
            "3",
            "--config",
            "BIGTIFF_OVERVIEW",
            "IF_SAFER",
            "-ro",
            "-r",
            "average",
            str(raster_path),
            *(str(level) for level in levels),
        ]
        # Run gdaladdo and convert a nonzero exit status to a Python exception.
        subprocess.run(command, check=True)
    # Verify the requested external sidecar instead of accepting internal overviews.
    if not overview_path.is_file() or overview_path.stat().st_size == 0:
        # Provide a targeted error when a GDAL build silently chose internal storage.
        raise RuntimeError(f"External overview was not created: {overview_path}")
    # Report the factors and sidecar location for ArcGIS users.
    print(f"External overview written: {overview_path} (levels: {levels})")
    # Return the verified sidecar path for the final summary.
    return overview_path


def output_path_for_orbit_direction(base_path: Path, orbit_direction: str) -> Path:
    """Return the output pathname dedicated to one validated orbit direction."""

    # Reject unrecognized labels before they become output filenames.
    if orbit_direction not in ORBIT_DIRECTIONS:
        raise ValueError(
            f"Unsupported orbit direction {orbit_direction!r}; "
            f"expected one of {ORBIT_DIRECTIONS}."
        )
    # Keep the configured name and extension while making the pass explicit.
    return base_path.with_name(
        f"{base_path.stem}_{orbit_direction}{base_path.suffix}"
    )


def main() -> None:
    """Run scene discovery, extraction, per-pass mosaicking, and pyramid creation."""

    # Resolve the configured output base to make direction-specific names absolute.
    output_base_path = OUTPUT_PATH.expanduser().resolve()
    # Restrict each direction-specific deliverable to a GeoTIFF filename.
    if output_base_path.suffix.lower() not in {".tif", ".tiff"}:
        # Prevent a driver/extension mismatch that can confuse ArcGIS.
        raise ValueError("OUTPUT_PATH must end with .tif or .tiff.")
    # Create the final output directory when it does not yet exist.
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    # Derive a clear default directory for retained per-scene extractions.
    extracted_dir = (
        EXTRACTED_DIR.expanduser().resolve()
        if EXTRACTED_DIR is not None
        else output_base_path.parent / f"{output_base_path.stem}_extracted"
    )
    # Create the extraction directory before opening any output rasters.
    extracted_dir.mkdir(parents=True, exist_ok=True)
    # Find every SME2 product in the input folder.
    scenes = discover_scenes(
        input_dir=INPUT_DIR.expanduser().resolve(),
        recursive=RECURSIVE,
    )
    # Refuse to make an empty mosaic when the folder holds no SME2 files.
    if not scenes:
        # Include the searched location in the actionable error.
        raise FileNotFoundError(f"No NISAR SME2 HDF5 files found in {INPUT_DIR}.")
    # Group every selected scene before extraction so opposite passes cannot
    # reach the same grid calculation or mosaicking loop.
    scenes_by_direction = {
        direction: [scene for scene in scenes if scene.orbit_direction == direction]
        for direction in ORBIT_DIRECTIONS
    }
    # Identify only output files that will actually be produced in this run.
    output_paths = {
        direction: output_path_for_orbit_direction(output_base_path, direction)
        for direction, scene_group in scenes_by_direction.items()
        if scene_group
    }
    # Protect every pending orbit-specific mosaic and sidecar before extraction.
    for direction, output_path in output_paths.items():
        if output_path.exists() and not OVERWRITE:
            raise FileExistsError(
                f"{direction} output exists; set OVERWRITE = True to replace it: "
                f"{output_path}"
            )
        existing_overview_path = Path(str(output_path) + ".ovr")
        if existing_overview_path.exists() and not OVERWRITE:
            raise FileExistsError(
                f"{direction} overview exists; set OVERWRITE = True to replace it: "
                f"{existing_overview_path}"
            )
    # Report the selected count and pass split before lengthy extraction begins.
    print(
        f"Selected {len(scenes)} SME2 scene(s): "
        + ", ".join(
            f"{direction}={len(scene_group)}"
            for direction, scene_group in scenes_by_direction.items()
            if scene_group
        )
    )
    # Extract independent HDF5 scenes concurrently, but never with more than
    # the conservative configured worker cap in memory at once.
    extracted_by_scene = extract_scenes_parallel(
        scenes=scenes,
        extracted_dir=extracted_dir,
        fallback_source_crs=SOURCE_CRS,
        overwrite=OVERWRITE,
        max_workers=MAX_EXTRACTION_WORKERS,
    )
    # Restore chronological order for deterministic metadata and processing even
    # though extraction itself completed in non-deterministic parallel order.
    extracted_paths_by_direction: dict[str, list[Path]] = {
        direction: [
            extracted_by_scene[scene.hdf5_path]
            for scene in scenes_by_direction[direction]
        ]
        for direction in ORBIT_DIRECTIONS
    }
    # Create a grid, mosaic, and overview independently for each orbit pass.
    completed_outputs: list[tuple[str, Path, Path]] = []
    for direction in ORBIT_DIRECTIONS:
        # Skip directions absent from the input scenes.
        if not scenes_by_direction[direction]:
            continue
        # Use only this direction's extractions when calculating the union grid.
        mosaic_grid = calculate_mosaic_grid(extracted_paths_by_direction[direction])
        # Mosaic only the scenes with the matching orbit direction, preserving
        # their native CRS and pixel values without resampling.
        output_path = output_paths[direction]
        # First mosaic in the native EASE-Grid so overlap averaging is performed
        # on aligned source pixels, then reproject only the completed mosaic.
        native_mosaic_path = output_path.with_name(output_path.stem + ".native.tif")
        native_mosaic_path.unlink(missing_ok=True)
        try:
            mosaic_extracted_rasters(
                extracted_paths=extracted_paths_by_direction[direction],
                output_path=native_mosaic_path,
                grid=mosaic_grid,
                scenes=scenes_by_direction[direction],
                overwrite=True,
            )
            reproject_mosaic_to_output_crs(
                source_path=native_mosaic_path,
                output_path=output_path,
                output_crs=OUTPUT_CRS,
                overwrite=OVERWRITE,
            )
        finally:
            # The native mosaic is temporary working state; final files use OUTPUT_CRS.
            native_mosaic_path.unlink(missing_ok=True)
        # Build and verify a separate ArcGIS-compatible external overview.
        overview_path = build_external_overviews(output_path, OVERWRITE)
        # Preserve the generated paths for the final concise run summary.
        completed_outputs.append((direction, output_path, overview_path))
    # Print every completed direction-specific deliverable.
    for direction, output_path, overview_path in completed_outputs:
        print(
            f"Complete. {direction} mosaic: {output_path}\n"
            f"Complete. {direction} pyramid: {overview_path}"
        )


# Run the workflow only when this file is executed, not when imported for tests.
if __name__ == "__main__":
    # Delegate all work to the documented main function.
    main()
