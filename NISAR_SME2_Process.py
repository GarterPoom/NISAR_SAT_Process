#!/usr/bin/env python3
r"""Export each NISAR SME2 soil-moisture raster to its own GeoTIFF.

The script performs the complete workflow requested for NISAR Level-3 SME2
products, one output per input product:

1. Find every NISAR SME2 HDF5 file in the input folder.
2. Extract only ``/science/LSAR/SME2/grids/soilMoisture`` and its coordinate /
   projection metadata to a native-grid GeoTIFF.
3. Exclude NaN, infinity, -9999, 0, and the HDF5 dataset's declared fill value.
   Those pixels remain nodata.
4. Reproject the native-grid GeoTIFF to the configured output CRS as
   ``<HDF5 product name>_soilMoisture.tif``.
5. Build internal nearest-neighbour pyramids inside each output GeoTIFF
   (no external ``.ovr`` file is created).

Products that already have both outputs are skipped unless ``OVERWRITE`` is
set, so new files can be added to the input folder and the script run again.

Configuration
-------------
Before running the script, edit ``INPUT_DIR`` and ``OUTPUT_DIR`` below. Then
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
# Copy the extraction when no reprojection is needed.
import shutil
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
# Reproject each native-grid raster to the requested deliverable CRS.
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Use the exact NISAR SME2 baseline-soil-moisture HDF5 dataset requested.
SOIL_MOISTURE_PATH = "/science/LSAR/SME2/grids/soilMoisture"
# Obtain coordinate and projection datasets from the parent grid group.
GRID_PATH = "/science/LSAR/SME2/grids"
# Read the acquisition pass from the standard NISAR product-identification group.
ORBIT_DIRECTION_PATH = "/science/LSAR/identification/orbitPassDirection"
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
# Folder for the final EPSG:4326 GeoTIFFs. Each one is named after its input
# product: ``<HDF5 product name>_soilMoisture.tif``.
OUTPUT_DIR = Path(r"soilMoisture")
# CRS of the final deliverables. Each scene is extracted on its native EASE-Grid
# first, then transformed to this CRS.
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
    # Store the end time so the output metadata records the full acquisition.
    acquisition_end: datetime
    # Keep orbit-pass metadata with the scene for the output's tags.
    orbit_direction: str


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
    """Read and normalize a scene's orbit-pass direction for its output metadata."""

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
    # Reject unexpected metadata rather than recording a wrong orbit pass.
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
        # Enable tiled access so ArcGIS display is efficient.
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

    Invalid cells are converted to ``-9999``. The output is
    normalized to a north-up, west-left orientation even when the HDF5 coordinate
    vectors are stored in the opposite direction.
    """

    # Give each extracted raster the unique full NISAR product stem.
    output_path = extracted_dir / f"{scene.hdf5_path.stem}_soilMoisture.tif"
    # Preserve an existing extraction unless replacement was explicitly requested.
    if output_path.exists() and not overwrite:
        # Report reuse so the result is not mistaken for a fresh extraction.
        print(f"Reusing existing extraction: {output_path}")
        # Return the existing path for reprojection.
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
            orbit_pass_direction=scene.orbit_direction,
            invalid_values="NaN, infinity, -9999, 0, and declared HDF5 fill values",
        )
        # Label the only band with the extracted NISAR variable name.
        destination.set_band_description(1, "soilMoisture")
    # Atomically replace the requested extraction only after a complete write.
    os.replace(partial_path, output_path)
    # Report the completed extraction for long multi-scene jobs.
    print(f"Extracted: {output_path}")
    # Return the extracted raster path to the reprojection stage.
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
    memory at any instant.
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


def reproject_to_output_crs(
    source_path: Path,
    output_path: Path,
    output_crs: str | CRS,
    overwrite: bool,
) -> None:
    """Reproject one native-grid raster into the final deliverable CRS.

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
            raise ValueError(f"Native raster has no CRS: {source_path}")
        if source.crs == target_crs:
            # Copy rather than move: the extraction is retained for later runs.
            shutil.copyfile(source_path, output_path)
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
    print(f"Reprojected raster written: {output_path} ({target_crs})")


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
    # Ensure even a small raster receives at least one overview level when possible.
    if not levels and min(width, height) >= 2:
        # A factor of 2 is the smallest useful overview.
        levels.append(2)
    # Return the factors for build_overviews.
    return levels


def build_internal_overviews(raster_path: Path, overwrite: bool) -> list[int]:
    """Build nearest-neighbour pyramids stored inside the GeoTIFF itself.

    No ``.ovr`` sidecar is produced.  Existing internal overviews are kept
    unless ``overwrite`` is set, in which case they are rebuilt.
    """

    # Read raster dimensions and any overviews already embedded in the file.
    with rasterio.open(raster_path) as source:
        # Calculate appropriate power-of-two decimation factors.
        levels = pyramid_levels(source.width, source.height)
        # Remember whether the file already carries internal overviews.
        existing = source.overviews(1)
    # Reject a one-pixel raster because no reduced-resolution pyramid can exist.
    if not levels:
        # Explain the physical limitation directly.
        raise ValueError(f"Raster is too small to build an overview: {raster_path}")
    # Leave an existing pyramid alone unless replacement was requested.
    if existing and not overwrite:
        print(f"Internal overviews already present: {raster_path} (levels: {existing})")
        return existing
    # Compress the embedded overview levels like the full-resolution band.
    with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW="3"):
        # Update mode appends the overviews inside the same .tif file.
        with rasterio.open(raster_path, "r+") as dataset:
            # Nearest neighbour keeps original values and the -9999 nodata unaltered.
            dataset.build_overviews(levels, Resampling.nearest)
    # Report the factors for the user.
    print(f"Internal overviews written: {raster_path} (levels: {levels})")
    # Return the factors that were built.
    return levels


def has_internal_overviews(raster_path: Path) -> bool:
    """Return True when the GeoTIFF already contains embedded overviews."""

    with rasterio.open(raster_path) as source:
        return bool(source.overviews(1))


def output_path_for_scene(output_dir: Path, scene: Scene) -> Path:
    """Return the GeoTIFF path for one scene, named after its HDF5 product."""

    # The full product name is unique and already records dates and orbit pass.
    return output_dir / f"{scene.hdf5_path.stem}_soilMoisture.tif"


def main() -> None:
    """Run scene discovery, extraction, reprojection, and pyramid creation."""

    # Resolve the output folder so reported paths are absolute.
    output_dir = OUTPUT_DIR.expanduser().resolve()
    # Create the final output directory when it does not yet exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Derive a clear default directory for retained per-scene extractions.
    extracted_dir = (
        EXTRACTED_DIR.expanduser().resolve()
        if EXTRACTED_DIR is not None
        else output_dir / "native_extracted"
    )
    # Create the extraction directory before opening any output rasters.
    extracted_dir.mkdir(parents=True, exist_ok=True)
    # Find every SME2 product in the input folder.
    scenes = discover_scenes(
        input_dir=INPUT_DIR.expanduser().resolve(),
        recursive=RECURSIVE,
    )
    # Stop with a clear message when the folder holds no SME2 files.
    if not scenes:
        # Include the searched location in the actionable error.
        raise FileNotFoundError(f"No NISAR SME2 HDF5 files found in {INPUT_DIR}.")
    # Leave finished products alone unless replacement was explicitly requested.
    pending_scenes = []
    for scene in scenes:
        output_path = output_path_for_scene(output_dir, scene)
        if not OVERWRITE and output_path.exists() and has_internal_overviews(output_path):
            print(f"Skipping existing output: {output_path}")
            continue
        pending_scenes.append(scene)
    # Report the workload before lengthy extraction begins.
    print(
        f"Found {len(scenes)} SME2 scene(s); "
        f"{len(pending_scenes)} to process."
    )
    # Extract independent HDF5 scenes concurrently, but never with more than
    # the conservative configured worker cap in memory at once.
    extracted_by_scene = extract_scenes_parallel(
        scenes=pending_scenes,
        extracted_dir=extracted_dir,
        fallback_source_crs=SOURCE_CRS,
        overwrite=OVERWRITE,
        max_workers=MAX_EXTRACTION_WORKERS,
    )
    # Create one reprojected GeoTIFF and overview for each scene.
    completed_outputs: list[tuple[Path, list[int]]] = []
    for scene in pending_scenes:
        output_path = output_path_for_scene(output_dir, scene)
        # A raster may already exist when an earlier run stopped before its
        # overview was built; in that case only the overview is still missing.
        if OVERWRITE or not output_path.exists():
            reproject_to_output_crs(
                source_path=extracted_by_scene[scene.hdf5_path],
                output_path=output_path,
                output_crs=OUTPUT_CRS,
                overwrite=OVERWRITE,
            )
        # Embed nearest-neighbour pyramids inside the GeoTIFF (no .ovr sidecar).
        levels = build_internal_overviews(output_path, OVERWRITE)
        # Preserve the generated paths for the final concise run summary.
        completed_outputs.append((output_path, levels))
    # Print every completed deliverable.
    for output_path, levels in completed_outputs:
        print(f"Complete. Raster with internal pyramid {levels}: {output_path}")


# Run the workflow only when this file is executed, not when imported for tests.
if __name__ == "__main__":
    # Delegate all work to the documented main function.
    main()
