#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create multi-temporal flood RGB GeoTIFFs from two NISAR HH scenes.

Purpose
-------
Read final HH dB rasters created by ``NISAR_Process.py`` in
``GeoTIFF_Processed`` and build the requested multi-temporal composite:
    Band 1 / Red   = non-flood or low-flood HH scene
    Band 2 / Green = high-flood HH scene
    Band 3 / Blue  = non-flood or low-flood HH scene

Scene matching
--------------
Track, frame, orbit direction, product type, processing mode, and frequency are
read from each NISAR filename. Only scenes whose matching identifiers are equal
and whose acquisition calendar dates differ can be combined. Acquisition dates
come from the source-product section of the filename, not the later GeoTIFF
export timestamp or filesystem dates.

Flood severity cannot be proven from a filename. By default, the earliest
available acquisition in each matching Track/Frame group is treated as the
non-flood or low-flood scene and the latest acquisition as the high-flood scene.
Use both ``--low-date`` and ``--high-date`` to select known event dates instead.

Input filename convention
-------------------------
``<NISAR_SOURCE>_<GSLC|GCOV>_<FREQUENCY>_HH_Processed_dB_<EXPORT>.tif``

The NISAR source name must contain a leading numeric identifier, Track,
direction, Frame, acquisition start, and acquisition end fields, for example:
``NISAR_L2_UR_GSLC_030_004_D_080_..._20260906T114638_20260906T114713_...``
In that example, Track is ``004``, direction is ``D``, and Frame is ``080``;
the preceding ``030`` field is not Track.
HV files, existing band stacks, and temporary ``.part.tif`` files are ignored.
When the same acquisition was exported more than once, the most recent export
timestamp is used so an older processing result is not selected accidentally.

Processing and outputs
----------------------
The two inputs must be single-band rasters on exactly the same CRS and pixel
grid; this script deliberately does not resample. Output values remain float32
dB values without a display stretch. A pixel is valid only where both dates are
valid and finite. Tiled I/O limits memory use, lossless DEFLATE compression
preserves values, and internal overviews improve map-display performance.

Default input:  ``GeoTIFF_Processed``
Default output: ``GeoTIFF_Processed\\Multi_Temporal_RGB``
Existing composites are skipped unless ``--overwrite`` is supplied.

Dependencies and usage
----------------------
Use the Python environment containing NumPy and Rasterio that runs
``NISAR_Process.py``::

    python NISAR_Flood_Multi_Temporal_RGB.py
    python NISAR_Flood_Multi_Temporal_RGB.py --low-date 20260801 --high-date 20260906
    python NISAR_Flood_Multi_Temporal_RGB.py --input-dir "GeoTIFF_Processed"
    python NISAR_Flood_Multi_Temporal_RGB.py --output-dir "NISAR\\Flood_RGB" --overwrite

In QGIS, choose Multiband color with Red=1, Green=2, and Blue=3, then apply a
per-band stretch suitable for the dB range.

Function guide
--------------
``parse_date_argument`` validates an optional command-line acquisition date.
``filename_info`` extracts Track, Frame, acquisition, and processing metadata.
``group_key`` returns the fields that must match across acquisition dates.
``choose_scene`` selects the newest export for one acquisition calendar date.
``discover_composites`` finds valid low/high HH date pairs for every group.
``validate_temporal_pair`` guards direct calls against mismatched filenames.
``stack_temporal_pair`` writes one aligned RGB composite tile by tile.
``main`` parses options, processes the batch, and reports the final counts.
"""

from __future__ import annotations  # Defer annotation evaluation for efficient imports.

import argparse  # Read input, output, date-selection, and overwrite command-line options.
from dataclasses import dataclass  # Store parsed filename fields in an immutable record.
from datetime import date, datetime  # Validate acquisition and export timestamps.
import logging  # Report discovery decisions, progress, warnings, and failures.
import os  # Atomically promote a finished temporary raster to its final name.
from pathlib import Path  # Handle platform-independent input and output paths.
import re  # Recognize processed GeoTIFFs and NISAR source-product identifiers.
import uuid  # Create a collision-resistant staging filename for each output.

import numpy as np  # Build shared masks and arrange the three output bands.
import rasterio  # Read and write georeferenced raster data and metadata.
from rasterio.enums import ColorInterp, Resampling  # Mark RGB roles and choose overview resampling.

DEFAULT_INPUT_DIRECTORY = Path(r"GeoTIFF_Processed")  # Point to the normal NISAR processing output folder.
DEFAULT_OUTPUT_SUBDIRECTORY = "Multi_Temporal_RGB"  # Keep composites separate from single-date products.
TILE_SIZE = 512  # Bound each read and write operation to a manageable pixel tile.
OUTPUT_NODATA = -9999.0  # Represent pixels invalid on either acquisition date.
PROCESSED_PATTERN = re.compile(  # Describe the final processed GeoTIFF filename suffix.
    r"^(?P<scene>.+)_(?P<product>GSLC|GCOV)_(?P<frequency>frequency[^_]+)_"  # Capture source and grid labels.
    r"(?P<pol>HH|HV)_Processed_dB_(?P<export_stamp>\d{8}_\d{6})\.tiff?$",  # Capture polarization and export time.
    re.IGNORECASE,  # Accept case variations while normalizing important fields later.
)  # Finish compiling the processed-filename pattern once at import time.
SOURCE_PATTERN = re.compile(  # Describe identifiers embedded in a standard NISAR source-product name.
    r"^NISAR_(?P<level>L\d+)_(?P<mode>[A-Z0-9]+)_(?P<source_product>GSLC|GCOV)_"  # Capture product family fields.
    r"\d+_(?P<track>\d+)_(?P<direction>[AD])_(?P<frame>\d+)_.*?"  # Skip leading ID, then capture Track/direction/Frame.
    r"(?P<acquisition_start>\d{8}T\d{6})_(?P<acquisition_end>\d{8}T\d{6})_.+$",  # Capture acquisition interval.
    re.IGNORECASE,  # Accept case variations found in otherwise valid product names.
)  # Finish compiling the source-product pattern once at import time.
LOG = logging.getLogger("NISAR_Flood_Multi_Temporal_RGB")  # Give every module message a consistent logger name.


@dataclass(frozen=True)  # Prevent accidental changes to identity fields after validation.
class SceneInfo:  # Define the complete parsed identity of one processed HH scene.
    """Hold filename-derived identity, acquisition, and export information.

    Attributes:
        path: Full path to the processed GeoTIFF.
        scene: Complete NISAR source-product name before the processing suffix.
        level: NISAR product level token, such as L2.
        mode: NISAR coverage or processing-mode token, such as UR.
        source_product: Product token embedded in the source name, such as GSLC.
        product: Product token written by ``NISAR_Process.py`` before frequency.
        frequency: Processed radar frequency label, such as frequencyA.
        polarization: Processed polarization; discovery accepts HH only.
        track: Zero-padded Track identifier from the NISAR source filename.
        frame: Zero-padded Frame identifier from the NISAR source filename.
        direction: Ascending (A) or descending (D) orbit direction.
        acquisition_start: Validated UTC-like acquisition start timestamp.
        acquisition_end: Validated UTC-like acquisition end timestamp.
        export_time: Validated timestamp for the processed GeoTIFF export run.
    """

    path: Path  # Retain the exact file selected for raster reading.
    scene: str  # Retain the full source name for provenance metadata.
    level: str  # Distinguish NISAR product processing levels.
    mode: str  # Distinguish source coverage or processing modes.
    source_product: str  # Preserve the product label embedded in the source name.
    product: str  # Preserve the product label in the processed filename suffix.
    frequency: str  # Prevent mixing frequencyA with another radar frequency.
    polarization: str  # Confirm that only HH is used for this composite.
    track: str  # Require the same satellite ground-track identifier.
    frame: str  # Require the same along-track frame identifier.
    direction: str  # Prevent pairing ascending and descending observations.
    acquisition_start: datetime  # Order scenes by the source acquisition time.
    acquisition_end: datetime  # Record the complete acquisition interval.
    export_time: datetime  # Prefer the latest processing run for duplicate acquisitions.


@dataclass(frozen=True)  # Keep every planned composite stable after discovery.
class Composite:  # Associate two selected scenes with their deterministic output stem.
    """Describe one low/high temporal pair ready for RGB stacking.

    Attributes:
        name: Filename stem for the output GeoTIFF.
        low: Earlier or explicitly selected non-flood/low-flood HH scene.
        high: Later or explicitly selected high-flood HH scene.
    """

    name: str  # Provide the output name without a filename extension.
    low: SceneInfo  # Supply both the red and blue channel source.
    high: SceneInfo  # Supply the green channel source.


def parse_date_argument(value: str) -> date:  # Define strict YYYYMMDD command-line parsing.
    """Convert a compact date argument to a validated calendar date.

    Args:
        value: Text in ``YYYYMMDD`` form, for example ``20260906``.
    Returns:
        The corresponding ``datetime.date`` value.
    Raises:
        argparse.ArgumentTypeError: The value has the wrong format or is not a
            real calendar date.
    """
    try:  # Convert and validate the complete eight-digit value in one operation.
        return datetime.strptime(value, "%Y%m%d").date()  # Return only the calendar-date portion.
    except ValueError as exc:  # Convert implementation detail into an argparse-friendly error.
        raise argparse.ArgumentTypeError("date must be a valid YYYYMMDD value") from exc  # Show concise command help.


def filename_info(path: Path) -> SceneInfo | None:  # Define parsing for one candidate directory entry.
    """Parse a final NISAR processed raster filename and validate all dates.

    Args:
        path: Candidate GeoTIFF path; only its filename is interpreted.
    Returns:
        A populated ``SceneInfo`` for a supported HH or HV processed raster, or
        ``None`` when the filename does not follow the required convention.
    Raises:
        ValueError: A matching filename has an impossible acquisition/export
            timestamp, a reversed acquisition interval, or inconsistent product
            tokens between its source name and processed suffix.
    """
    processed_match = PROCESSED_PATTERN.fullmatch(path.name)  # Match the entire final processed filename.

    if processed_match is None:  # Detect unrelated rasters, folders, and temporary products.
        return None  # Tell discovery to ignore unsupported entries safely.

    processed = processed_match.groupdict()  # Extract the outer processed-product fields by name.
    source_match = SOURCE_PATTERN.fullmatch(processed["scene"])  # Parse the embedded NISAR source filename.

    if source_match is None:  # Require standard source fields so Track and Frame are trustworthy.
        return None  # Ignore non-NISAR or shortened names instead of guessing field positions.

    source = source_match.groupdict()  # Extract source-product fields into an accessible mapping.

    acquisition_start = datetime.strptime(source["acquisition_start"], "%Y%m%dT%H%M%S")  # Validate acquisition start.
    acquisition_end = datetime.strptime(source["acquisition_end"], "%Y%m%dT%H%M%S")  # Validate acquisition end.
    export_time = datetime.strptime(processed["export_stamp"], "%Y%m%d_%H%M%S")  # Validate processing export time.

    if acquisition_end < acquisition_start:  # Reject a malformed interval before using its date.
        raise ValueError(f"Acquisition end precedes start in {path.name}")  # Identify the invalid input precisely.

    source_product = source["source_product"].upper()  # Normalize the embedded product token for comparison.
    product = processed["product"].upper()  # Normalize the processed suffix product token for comparison.

    if source_product != product:  # Detect inconsistent GSLC/GCOV identity within one filename.
        raise ValueError(f"Source and processed product types differ in {path.name}")  # Refuse uncertain grouping.

    return SceneInfo(  # Assemble one immutable, fully validated scene record.
        path=path,  # Store the file to open when writing a selected composite.
        scene=processed["scene"],  # Preserve the complete source product name.
        level=source["level"].upper(),  # Normalize the NISAR processing level token.
        mode=source["mode"].upper(),  # Normalize the source processing-mode token.
        source_product=source_product,  # Store the validated source product family.
        product=product,  # Store the validated processed product family.
        frequency=processed["frequency"],  # Preserve the processed frequency spelling for names and metadata.
        polarization=processed["pol"].upper(),  # Normalize HH/HV before channel filtering.
        track=source["track"],  # Preserve zero padding in the Track identifier.
        frame=source["frame"],  # Preserve zero padding in the Frame identifier.
        direction=source["direction"].upper(),  # Normalize the orbit-direction code.
        acquisition_start=acquisition_start,  # Store the timestamp used for temporal ordering.
        acquisition_end=acquisition_end,  # Store the acquisition interval endpoint for provenance.
        export_time=export_time,  # Store the timestamp used to resolve repeated processing runs.
    )  # Finish constructing the parsed scene record.


def group_key(info: SceneInfo) -> tuple[str, ...]:  # Define the identity shared by valid temporal partners.
    """Return the non-temporal fields that two acquisition dates must share.

    Args:
        info: Parsed scene metadata from ``filename_info``.
    Returns:
        A tuple containing level, mode, product identities, frequency, Track,
        Frame, and orbit direction in deterministic order.
    """
    return (  # Build a hashable key suitable for a dictionary.
        info.level,  # Require the same NISAR processing level.
        info.mode,  # Require the same source coverage or processing mode.
        info.source_product,  # Require the same embedded product family.
        info.product,  # Require the same processed product family.
        info.frequency.lower(),  # Match frequency labels without case sensitivity.
        info.track,  # Enforce the user's same-Track requirement.
        info.frame,  # Enforce the user's same-Frame requirement.
        info.direction,  # Avoid combining opposite viewing geometries.
    )  # Finish the temporal matching key.


def choose_scene(candidates: list[SceneInfo], acquisition_date: date) -> SceneInfo:  # Resolve one date's candidates.
    """Select the newest export for one group and acquisition calendar date.

    Args:
        candidates: HH scenes sharing a matching key and acquisition date.
        acquisition_date: Date being resolved, used in diagnostic messages.
    Returns:
        The candidate with the latest GeoTIFF export timestamp.
    Raises:
        ValueError: Multiple different acquisitions exist on the requested date,
            because a calendar-date option cannot select between them safely.
    """
    acquisition_times = {item.acquisition_start for item in candidates}  # Find distinct passes on this date.

    if len(acquisition_times) != 1:  # Detect an ambiguity that export timestamps cannot resolve.
        names = ", ".join(item.path.name for item in sorted(candidates, key=lambda item: item.path.name))  # List evidence.
        raise ValueError(f"Multiple HH acquisitions exist on {acquisition_date:%Y%m%d}: {names}")  # Refuse to guess.

    selected = max(candidates, key=lambda item: (item.export_time, item.path.name))  # Prefer the newest processing result.

    if len(candidates) > 1:  # Make automatic duplicate resolution visible to the operator.
        LOG.warning(  # Report the chosen file and number of older alternatives.
            "Using newest export for %s Track %s Frame %s: %s (%d older export(s) ignored)",  # Define message fields.
            acquisition_date.isoformat(),  # Show the source acquisition calendar date.
            selected.track,  # Show the matching Track identifier.
            selected.frame,  # Show the matching Frame identifier.
            selected.path.name,  # Identify the actual input selected.
            len(candidates) - 1,  # Count the ignored older exports.
        )  # Finish the duplicate-resolution warning.

    return selected  # Supply the unique newest export to composite discovery.


def discover_composites(  # Define directory scanning and temporal pairing.
    input_dir: Path,  # Receive the directory containing processed rasters.
    low_date: date | None = None,  # Optionally require a known low-flood acquisition date.
    high_date: date | None = None,  # Optionally require a known high-flood acquisition date.
) -> tuple[list[Composite], int]:  # Return complete plans and a count of unusable groups.
    """Find same-Track/Frame HH pairs acquired on different calendar dates.

    Args:
        input_dir: Directory containing final GeoTIFFs from ``NISAR_Process.py``.
        low_date: Optional known non-flood/low-flood acquisition calendar date.
        high_date: Optional known high-flood acquisition calendar date.
    Returns:
        ``(composites, incomplete_count)``. With no dates, each compatible group
        uses its earliest and latest dates. With dates, each group must contain
        both exact dates. Groups with fewer than two dates or missing requested
        dates are reported and counted as incomplete.
    Raises:
        ValueError: Only one date option is provided, dates are equal/reversed,
            a recognized filename is invalid, or same-day acquisitions conflict.
        OSError: The input directory cannot be read.
    """
    if (low_date is None) != (high_date is None):  # Require date overrides as one meaningful pair.
        raise ValueError("--low-date and --high-date must be supplied together")  # Prevent a half-defined event.

    if low_date is not None and high_date is not None and low_date >= high_date:  # Require different chronological dates.
        raise ValueError("--low-date must be earlier than --high-date")  # Keep channel roles temporally consistent.

    groups: dict[tuple[str, ...], dict[date, list[SceneInfo]]] = {}  # Index matching groups, then acquisition dates.

    for path in sorted(input_dir.iterdir()):  # Inspect entries in deterministic filename order.
        if not path.is_file():  # Exclude output folders and any other directories.
            continue  # Advance to the next entry without attempting filename parsing.

        info = filename_info(path)  # Parse and validate any supported processed raster filename.

        if info is None or info.polarization != "HH":  # Use only HH, ignoring HV and unrelated products.
            continue  # Advance without treating intentionally ignored inputs as errors.

        by_date = groups.setdefault(group_key(info), {})  # Retrieve this Track/Frame-compatible group.
        by_date.setdefault(info.acquisition_start.date(), []).append(info)  # Register the scene under its source date.

    composites: list[Composite] = []  # Collect valid output plans in deterministic group order.
    incomplete = 0  # Count groups that cannot satisfy the requested temporal pairing.

    for key, by_date in sorted(groups.items()):  # Resolve every compatible Track/Frame group independently.
        level, mode, source_product, product, frequency_key, track, frame, direction = key  # Unpack name fields.
        available_dates = sorted(by_date)  # Order acquisitions using filename timestamps, never file dates.

        if low_date is None or high_date is None:  # Apply automatic earliest/latest selection.
            if len(available_dates) < 2:  # Require two distinct acquisition calendar dates.
                incomplete += 1  # Record that this Track/Frame group cannot be processed yet.
                LOG.warning(  # Explain the exact group and its only available dates.
                    "Skipping Track %s Frame %s %s: need two HH dates; found %s",  # Define the diagnostic format.
                    track,  # Identify the unusable Track.
                    frame,  # Identify the unusable Frame.
                    direction,  # Identify the orbit direction.
                    ", ".join(item.strftime("%Y%m%d") for item in available_dates) or "none",  # List source dates.
                )  # Finish the missing-date warning.
                continue  # Advance to another group without creating an invalid composite.

            selected_low_date = available_dates[0]  # Treat the earliest date as low flood by documented default.
            selected_high_date = available_dates[-1]  # Treat the latest date as high flood by documented default.
        else:  # Apply the operator's known flood-event date choices.
            selected_low_date = low_date  # Use the explicit red/blue acquisition date.
            selected_high_date = high_date  # Use the explicit green acquisition date.
            missing_dates = [item for item in (selected_low_date, selected_high_date) if item not in by_date]  # Check both.

            if missing_dates:  # Skip a group that does not contain the complete requested pair.
                incomplete += 1  # Include the group in the partial-failure exit status.
                LOG.warning(  # Report requested, missing, and available dates for correction.
                    "Skipping Track %s Frame %s %s: missing requested HH date(s) %s; available %s",  # Define fields.
                    track,  # Identify the affected Track.
                    frame,  # Identify the affected Frame.
                    direction,  # Identify the affected orbit direction.
                    ", ".join(item.strftime("%Y%m%d") for item in missing_dates),  # List unavailable selections.
                    ", ".join(item.strftime("%Y%m%d") for item in available_dates),  # List usable source dates.
                )  # Finish the requested-date warning.
                continue  # Advance without silently substituting a different date.

        low = choose_scene(by_date[selected_low_date], selected_low_date)  # Resolve the newest low-date export.
        high = choose_scene(by_date[selected_high_date], selected_high_date)  # Resolve the newest high-date export.
        frequency = low.frequency  # Preserve original frequency capitalization in the output filename.

        name = (  # Build a readable, collision-resistant output filename stem.
            f"NISAR_Track{track}_Frame{frame}_{direction}_{level}_{mode}_{source_product}_"  # Record matching geometry.
            f"{product}_{frequency}_MultiTemporal_RGB_{selected_low_date:%Y%m%d}_{selected_high_date:%Y%m%d}"  # Record dates.
        )  # Finish the output name.

        composites.append(Composite(name=name, low=low, high=high))  # Store the validated pair for processing.

    return composites, incomplete  # Return all complete plans and skipped-group count.


def validate_temporal_pair(low_path: Path, high_path: Path) -> tuple[SceneInfo, SceneInfo]:  # Guard direct use.
    """Verify that two paths are compatible HH scenes from different dates.

    Args:
        low_path: Non-flood or low-flood processed HH GeoTIFF path.
        high_path: High-flood processed HH GeoTIFF path.
    Returns:
        The parsed low and high ``SceneInfo`` records.
    Raises:
        ValueError: Either filename is unsupported, either polarization is not
            HH, Track/Frame or another group field differs, or acquisition dates
            are equal/reversed.
    """
    low = filename_info(low_path)  # Parse and validate the proposed red/blue source.
    high = filename_info(high_path)  # Parse and validate the proposed green source.

    if low is None or high is None:  # Require filenames with reliable NISAR identity fields.
        raise ValueError("Both inputs must follow the final NISAR processed GeoTIFF naming convention")  # Reject guesses.

    if low.polarization != "HH" or high.polarization != "HH":  # Enforce the requested HH-only band mapping.
        raise ValueError("Both temporal inputs must be HH rasters")  # Reject HV or swapped product inputs.

    if group_key(low) != group_key(high):  # Compare Track, Frame, direction, product, mode, and frequency.
        raise ValueError("Temporal inputs must share Track, Frame, direction, level, mode, product, and frequency")  # Explain.

    if low.acquisition_start.date() >= high.acquisition_start.date():  # Require low scene to precede high scene by date.
        raise ValueError("Low-flood acquisition date must be earlier than high-flood acquisition date")  # Enforce roles.

    return low, high  # Supply validated source metadata to the raster writer.


def stack_temporal_pair(low_path: Path, high_path: Path, output_path: Path) -> None:  # Define RGB writing.
    """Write one aligned low/high HH pair as a georeferenced RGB float32 stack.

    Args:
        low_path: Single-band non-flood or low-flood HH raster in dB.
        high_path: Single-band high-flood HH raster in dB.
        output_path: Destination GeoTIFF whose parent directory already exists.
    Returns:
        ``None``. Writes R=low HH, G=high HH, and B=low HH with provenance.
    Raises:
        ValueError: Filename identities, acquisition dates, band counts, spatial
            grids, CRS information, or output paths are incompatible.
        OSError or rasterio.errors.RasterioError: Raster reading/writing fails.
    Notes:
        Both dates share one validity mask. The final filename appears only after
        all pixels, metadata, mask, and overviews have been written successfully.
        This function can replace an output; ``main`` enforces ``--overwrite``.
    """
    low_info, high_info = validate_temporal_pair(low_path, high_path)  # Recheck identity for direct function calls.
    resolved_output = output_path.resolve()  # Normalize the destination before comparing it with sources.

    if resolved_output in {low_path.resolve(), high_path.resolve()}:  # Protect both input datasets from replacement.
        raise ValueError("Output must not replace either input raster")  # Stop before a writer can truncate a source.

    temporary = output_path.with_name(f".{uuid.uuid4().hex}.part.tif")  # Allocate a unique staging raster nearby.

    try:  # Guarantee cleanup of this call's incomplete staging file.
        with rasterio.open(low_path) as low, rasterio.open(high_path) as high:  # Open and later close both source rasters.
            if low.count != 1 or high.count != 1:  # Require original single-band processed products.
                raise ValueError("Expected single-band HH inputs")  # Reject a prior stack or unexpected dataset.

            if low.crs is None or high.crs is None:  # Require defined coordinate reference systems.
                raise ValueError("Both inputs must have a coordinate reference system")  # Prevent ungeoreferenced output.

            if low.crs != high.crs or low.shape != high.shape or low.transform != high.transform:  # Compare full grids.
                raise ValueError("Temporal HH grids differ; align CRS, dimensions, and transform before stacking")  # Reject.

            profile = {  # Define storage, band structure, compression, and georeferencing.
                "driver": "GTiff",  # Write the result using the GeoTIFF driver.
                "width": low.width,  # Preserve the shared number of pixel columns.
                "height": low.height,  # Preserve the shared number of pixel rows.
                "count": 3,  # Allocate the requested red, green, and blue bands.
                "dtype": "float32",  # Retain continuous negative and fractional dB values.
                "crs": low.crs,  # Preserve the shared coordinate reference system.
                "transform": low.transform,  # Preserve origin, pixel size, rotation, and orientation.
                "nodata": OUTPUT_NODATA,  # Declare a common invalid-pixel value for all bands.
                "tiled": True,  # Store data in tiles for efficient windowed access.
                "blockxsize": TILE_SIZE,  # Use the configured tile width.
                "blockysize": TILE_SIZE,  # Use the configured tile height.
                "compress": "deflate",  # Apply lossless compression to preserve measurements.
                "predictor": 3,  # Improve compression efficiency for floating-point samples.
                "BIGTIFF": "IF_SAFER",  # Permit BigTIFF automatically when output size requires it.
                "photometric": "RGB",  # Advertise the bands as a true RGB visualization mapping.
                "interleave": "pixel",  # Store the three channel values together per pixel.
            }  # Finish the destination profile.

            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):  # Embed validity in the TIFF rather than a sidecar file.
                with rasterio.open(temporary, "w", **profile) as dst:  # Create and safely close the staging dataset.
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)  # Declare display roles.

                    low_date_text = low_info.acquisition_start.strftime("%Y-%m-%d")  # Format the low date once.
                    high_date_text = high_info.acquisition_start.strftime("%Y-%m-%d")  # Format the high date once.

                    band_labels = (  # Define self-explanatory descriptions in exact output order.
                        f"HH non-flood/low-flood ({low_date_text})",  # Describe the red-band source and date.
                        f"HH high-flood ({high_date_text})",  # Describe the green-band source and date.
                        f"HH non-flood/low-flood ({low_date_text})",  # Describe the repeated blue-band source.
                    )  # Finish band-label construction.

                    for index, label in enumerate(band_labels, 1):  # Number Rasterio bands from one.
                        dst.set_band_description(index, label)  # Embed the human-readable role on each channel.
                        dst.set_band_unit(index, "dB")  # Record the measurement unit for every output channel.

                    dst.update_tags(  # Record provenance and matching evidence at dataset level.
                        SOURCE_LOW_HH=low_path.name,  # Identify the red/blue source file exactly.
                        SOURCE_HIGH_HH=high_path.name,  # Identify the green source file exactly.
                        TRACK=low_info.track,  # Record the verified common Track identifier.
                        FRAME=low_info.frame,  # Record the verified common Frame identifier.
                        ORBIT_DIRECTION=low_info.direction,  # Record the verified common viewing direction.
                        PRODUCT_TYPE=low_info.product,  # Record the verified common processed product family.
                        FREQUENCY=low_info.frequency,  # Record the verified common radar frequency.
                        LOW_ACQUISITION_START=low_info.acquisition_start.isoformat(),  # Record low-scene start time.
                        LOW_ACQUISITION_END=low_info.acquisition_end.isoformat(),  # Record low-scene end time.
                        HIGH_ACQUISITION_START=high_info.acquisition_start.isoformat(),  # Record high-scene start time.
                        HIGH_ACQUISITION_END=high_info.acquisition_end.isoformat(),  # Record high-scene end time.
                        LOW_EXPORT_TIMESTAMP=low_info.export_time.isoformat(),  # Record low-scene processing run.
                        HIGH_EXPORT_TIMESTAMP=high_info.export_time.isoformat(),  # Record high-scene processing run.
                        PAIRING_RULE="Same Track, Frame, direction, level, mode, product, and frequency; different dates",  # Explain.
                        BAND_MAPPING="R=low/non-flood HH; G=high-flood HH; B=low/non-flood HH",  # Document channels.
                        VALIDITY="All bands valid only where both temporal HH source pixels are valid",  # Explain mask.
                    )  # Finish writing dataset-level metadata.

                    tile_rows = (low.height + TILE_SIZE - 1) // TILE_SIZE  # Round up the output tile-row count.
                    tile_columns = (low.width + TILE_SIZE - 1) // TILE_SIZE  # Round up the tile-column count.
                    total = tile_rows * tile_columns  # Count all windows for progress percentages.

                    for number, (_, window) in enumerate(dst.block_windows(1), 1):  # Visit every output tile once.
                        low_data = low.read(1, window=window, masked=True, out_dtype="float32")  # Read low HH and mask.
                        high_data = high.read(1, window=window, masked=True, out_dtype="float32")  # Read high HH and mask.

                        valid = ~np.ma.getmaskarray(low_data) & ~np.ma.getmaskarray(high_data)  # Require both masks valid.
                        valid &= np.isfinite(low_data.data) & np.isfinite(high_data.data)  # Exclude NaN and infinity.

                        data = np.stack((low_data.data, high_data.data, low_data.data)).astype(np.float32, copy=False)  # Map RGB.
                        data[:, ~valid] = OUTPUT_NODATA  # Apply the identical invalid footprint to all three channels.

                        dst.write(data, window=window)  # Write this completed three-band tile.
                        dst.write_mask(valid.astype(np.uint8) * 255, window=window)  # Encode valid=255 and invalid=0.

                        if number == total or number % max(1, total // 10) == 0:  # Report about every ten percent.
                            LOG.info("Stacking: %.0f%% (%d/%d tiles)", 100 * number / total, number, total)  # Show progress.

                    factors = [factor for factor in (2, 4, 8, 16, 32) if min(low.width, low.height) // factor >= 1]  # Fit.

                    if factors:  # Avoid invalid overviews for extremely small rasters.
                        dst.build_overviews(factors, Resampling.average)  # Build continuous-data display pyramids.
                        dst.update_tags(ns="rio_overview", resampling="average")  # Record the overview algorithm.

        os.replace(temporary, output_path)  # Publish only the fully closed and completed raster.

    finally:  # Run after success and every possible read/write exception.
        if temporary.exists():  # Detect whether this call left an incomplete staging file.
            temporary.unlink()  # Remove only the uniquely named temporary output.


def main() -> int:  # Define the command-line batch workflow.
    """Discover selected temporal pairs, create RGB files, and report outcomes.

    Args:
        None. Reads command-line options through ``argparse``.
    Returns:
        ``0`` when all discovered groups are created or already exist; ``1``
        when input discovery fails, no valid pair exists, a group is incomplete,
        or an individual composite fails.
    Side effects:
        Creates the output directory, writes GeoTIFF composites, and logs status.
        A failed group is reported while remaining independent groups continue.
    """
    parser = argparse.ArgumentParser(  # Configure comprehensive command-line help.
        description=__doc__,  # Reuse the module documentation as the workflow description.
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve documentation formatting.
    )  # Finish parser construction.

    parser.add_argument(  # Define the processed-raster source folder.
        "--input-dir",  # Expose a readable long option name.
        type=Path,  # Convert text to a platform-aware path.
        default=DEFAULT_INPUT_DIRECTORY,  # Use the normal processing output by default.
        help="Folder containing final processed HH GeoTIFFs",  # Explain the accepted folder content.
    )  # Finish the input-folder option.

    parser.add_argument(  # Define an optional composite destination override.
        "--output-dir",  # Expose a readable long option name.
        type=Path,  # Convert supplied text to a platform-aware path.
        help="Default: INPUT_DIR/Multi_Temporal_RGB",  # Explain the automatic destination.
    )  # Finish the output-folder option.

    parser.add_argument(  # Define selection of a known non-flood/low-flood date.
        "--low-date",  # Use terminology matching the requested channel mapping.
        type=parse_date_argument,  # Validate the compact calendar date immediately.
        help="Non-flood/low-flood acquisition date (YYYYMMDD); requires --high-date",  # Explain pairing.
    )  # Finish the low-date option.

    parser.add_argument(  # Define selection of a known high-flood date.
        "--high-date",  # Use terminology matching the requested channel mapping.
        type=parse_date_argument,  # Validate the compact calendar date immediately.
        help="High-flood acquisition date (YYYYMMDD); requires --low-date",  # Explain pairing.
    )  # Finish the high-date option.

    parser.add_argument(  # Define explicit replacement of completed composites.
        "--overwrite",  # Expose a readable boolean switch.
        action="store_true",  # Default to false and become true when specified.
        help="Replace existing composites; otherwise skip them",  # Explain the safe default.
    )  # Finish the overwrite option.

    args = parser.parse_args()  # Parse and validate all command-line values.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")  # Configure messages.

    if not args.input_dir.is_dir():  # Check the source before creating any output folder.
        LOG.error("Input directory does not exist: %s", args.input_dir)  # Identify the missing folder.
        return 1  # Signal failure to the calling shell.

    try:  # Convert discovery exceptions into a concise batch failure.
        composites, incomplete = discover_composites(args.input_dir, args.low_date, args.high_date)  # Find pairs.
    except (ValueError, OSError) as exc:  # Catch invalid names, date choices, ambiguity, or unreadable folders.
        LOG.error("Composite discovery failed: %s", exc)  # Explain why processing could not begin.
        return 1  # Stop before creating potentially mismatched products.

    if not composites:  # Detect a folder with no usable two-date HH group.
        LOG.error("No same-Track/Frame HH scenes on two different acquisition dates were found")  # Explain.
        return 1  # Signal that no requested output could be produced.

    output_dir = args.output_dir or args.input_dir / DEFAULT_OUTPUT_SUBDIRECTORY  # Choose requested or default output.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the destination and missing parent folders.
    LOG.info("Found %d multi-temporal pair(s)", len(composites))  # Report the batch size before raster I/O.

    created = skipped = failed = 0  # Initialize complete, retained, and unsuccessful output counts.

    for composite in composites:  # Process each independently matched Track/Frame group.
        output_path = output_dir / f"{composite.name}.tif"  # Add the GeoTIFF extension to the planned stem.

        if output_path.exists() and not args.overwrite:  # Preserve prior results unless replacement was requested.
            LOG.info("Already exists, skipping: %s", output_path)  # Identify the retained composite.
            skipped += 1  # Count the existing output.
            continue  # Advance without opening or changing the existing file.

        try:  # Isolate raster failures to one Track/Frame pair.
            LOG.info("Low/non-flood HH: %s", composite.low.path.name)  # Show the red/blue source selection.
            LOG.info("High-flood HH: %s", composite.high.path.name)  # Show the green source selection.
            stack_temporal_pair(composite.low.path, composite.high.path, output_path)  # Validate and write RGB.
            created += 1  # Count the successfully published composite.
            LOG.info("Created: %s", output_path)  # Report its final destination.
        except Exception:  # Continue other independent groups after an unexpected raster error.
            failed += 1  # Count the unsuccessful composite.
            LOG.exception("Failed composite: %s", composite.name)  # Include a diagnostic traceback.

    LOG.info("Done: %d created, %d existing, %d incomplete, %d failed", created, skipped, incomplete, failed)  # Summarize.

    return 1 if failed or incomplete else 0  # Signal partial failure whenever a discovered group was not handled.


if __name__ == "__main__":  # Run the batch only when this file is executed, not when imported.
    raise SystemExit(main())  # Return the workflow result as the process exit status.
