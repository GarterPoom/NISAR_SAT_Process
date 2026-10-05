#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create multi-temporal flood RGB GeoTIFFs with HV red/blue and HH green.

This is a documented companion to ``NISAR_Flood_Multi_Temporal_RGB.py``.
It retains the original temporal selection, filename validation, tiled I/O,
common validity mask, compression, metadata, and overview behaviour, but maps
the polarization channels as follows:

    * Red:  HV non-flood or low-flood acquisition.
    * Green: HH high-flood acquisition (the same green-channel role as before).
    * Blue: HV non-flood or low-flood acquisition.

The earliest available HV date is used as the low-flood date and the latest HH
date after it is used as the high-flood date.  Use ``--low-date`` and
``--high-date`` together when the flood dates are known.  A pair is accepted
only when its products have matching level, mode, product, frequency, Track,
Frame, and orbit direction; their source dates must differ.  The two rasters
must also have exactly the same CRS, shape, and transform because this module
does not resample scientific measurements.

Input names must follow the final processed-raster convention produced by
``NISAR_Process.py``::

    <NISAR_SOURCE>_<GSLC|GCOV>_<FREQUENCY>_<HH|HV>_Processed_dB_<EXPORT>.tif

HV is intentionally required for red/blue and HH is intentionally required for
green.  Other polarizations, band stacks, temporary ``.part.tif`` files, and
unrecognised filenames are ignored.

Examples::

    python NISAR_Flood_Multi_Temporal_HV_HH_RGB.py
    python NISAR_Flood_Multi_Temporal_HV_HH_RGB.py --low-date 20260801 --high-date 20260906
    python NISAR_Flood_Multi_Temporal_HV_HH_RGB.py --input-dir GeoTIFF_Processed --overwrite

The output folder defaults to ``GeoTIFF_Processed/Multi_Temporal_HV_HH_RGB``.
In QGIS, select Multiband color with Red=1, Green=2, and Blue=3, then choose a
display stretch appropriate for the dB range.

Function guide
--------------
``parse_date_argument`` validates command-line dates.
``filename_info`` extracts and verifies NISAR filename metadata.
``group_key`` returns metadata that must agree between source products.
``choose_scene`` resolves duplicate exports for one date and polarization.
``discover_composites`` plans valid HV-low/HH-high temporal pairs.
``validate_temporal_pair`` protects direct raster-writing calls.
``stack_temporal_pair`` creates a tile-by-tile RGB GeoTIFF.
``main`` provides the command-line batch workflow.
"""

from __future__ import annotations  # Defer annotation evaluation until it is needed.

import argparse  # Parse command-line paths, dates, and overwrite choices.
from dataclasses import dataclass  # Define immutable parsed-file records.
from datetime import date, datetime  # Validate source and export timestamps.
import logging  # Report selection decisions, progress, and failures.
import os  # Atomically promote finished temporary outputs.
from pathlib import Path  # Work safely with input and output filesystem paths.
import re  # Parse processed raster and embedded NISAR source names.
import uuid  # Generate unique temporary filenames beside final outputs.

import numpy as np  # Construct RGB arrays and shared validity masks.
import rasterio  # Read and write georeferenced raster datasets.
from rasterio.enums import ColorInterp, Resampling  # Declare RGB roles and overview method.

DEFAULT_INPUT_DIRECTORY = Path("GeoTIFF_Processed")  # Define the normal processed-raster location.
DEFAULT_OUTPUT_SUBDIRECTORY = "Multi_Temporal_HV_HH_RGB"  # Keep this channel variant separate.
TILE_SIZE = 512  # Limit windowed reads and writes to practical tile dimensions.
OUTPUT_NODATA = -9999.0  # Store pixels invalid in either source as one float value.

PROCESSED_PATTERN = re.compile(  # Match the complete final GeoTIFF naming suffix.
    r"^(?P<scene>.+)_(?P<product>GSLC|GCOV)_(?P<frequency>frequency[^_]+)_"  # Capture source/product/frequency.
    r"(?P<pol>HH|HV)_Processed_dB_(?P<export_stamp>\d{8}_\d{6})\.tiff?$",  # Capture polarization/export time.
    re.IGNORECASE,  # Permit harmless filename case variations.
)  # Compile the processed filename expression once at module import.

SOURCE_PATTERN = re.compile(  # Match metadata embedded in the NISAR source product name.
    r"^NISAR_(?P<level>L\d+)_(?P<mode>[A-Z0-9]+)_(?P<source_product>GSLC|GCOV)_"  # Capture product family.
    r"\d+_(?P<track>\d+)_(?P<direction>[AD])_(?P<frame>\d+)_.*?"  # Skip ID and capture geometry.
    r"(?P<acquisition_start>\d{8}T\d{6})_(?P<acquisition_end>\d{8}T\d{6})_.+$",  # Capture source interval.
    re.IGNORECASE,  # Permit compatible case variations.
)  # Compile the source filename expression once at module import.

LOG = logging.getLogger("NISAR_Flood_Multi_Temporal_HV_HH_RGB")  # Use one named module logger.


@dataclass(frozen=True)  # Prevent later mutation of validated filename identity.
class SceneInfo:
    """Hold the identity, acquisition interval, and export time of one raster.

    Attributes:
        path: Exact processed GeoTIFF file selected for reading.
        scene: Complete NISAR source name before the processed suffix.
        level: NISAR product level, such as ``L2``.
        mode: NISAR coverage or processing-mode token.
        source_product: Product type from the embedded source name.
        product: Product type from the processed filename suffix.
        frequency: Processed radar-frequency label.
        polarization: Normalized ``HH`` or ``HV`` polarization code.
        track: Zero-padded NISAR Track identifier.
        frame: Zero-padded NISAR Frame identifier.
        direction: Ascending ``A`` or descending ``D`` orbit direction.
        acquisition_start: Validated source acquisition start timestamp.
        acquisition_end: Validated source acquisition end timestamp.
        export_time: Validated processed-raster export timestamp.
    """

    path: Path  # Retain the file path used by Rasterio.
    scene: str  # Retain the original source product name for provenance.
    level: str  # Distinguish processing levels during matching.
    mode: str  # Distinguish coverage or processing modes during matching.
    source_product: str  # Preserve the embedded GSLC or GCOV product family.
    product: str  # Preserve the processed filename's GSLC or GCOV family.
    frequency: str  # Prevent mixing different radar-frequency outputs.
    polarization: str  # Identify the required HV or HH channel source.
    track: str  # Require a common satellite ground track.
    frame: str  # Require a common along-track frame.
    direction: str  # Avoid combining ascending and descending observations.
    acquisition_start: datetime  # Order sources by acquisition date.
    acquisition_end: datetime  # Preserve full temporal provenance.
    export_time: datetime  # Resolve repeated exports of one acquisition.


@dataclass(frozen=True)  # Keep a discovered processing plan immutable.
class Composite:
    """Describe one HV-low/HH-high RGB composite ready for creation.

    Attributes:
        name: Deterministic output filename stem without ``.tif``.
        low_hv: HV source used for output red and blue bands.
        high_hh: HH source used for output green band.
    """

    name: str  # Store the output stem.
    low_hv: SceneInfo  # Supply red and blue source data.
    high_hh: SceneInfo  # Supply green source data.


def parse_date_argument(value: str) -> date:
    """Convert a strict ``YYYYMMDD`` command-line value to ``date``.

    Args:
        value: Eight-digit acquisition date supplied by the operator.
    Returns:
        The corresponding validated calendar date.
    Raises:
        argparse.ArgumentTypeError: The text is malformed or not a real date.
    """
    try:  # Parse the complete value and validate its calendar components.
        return datetime.strptime(value, "%Y%m%d").date()  # Return only the date part.
    except ValueError as exc:  # Convert the low-level parsing error into CLI help.
        raise argparse.ArgumentTypeError("date must be a valid YYYYMMDD value") from exc  # Explain required format.


def filename_info(path: Path) -> SceneInfo | None:
    """Parse a final processed NISAR raster filename into validated metadata.

    Args:
        path: Candidate path whose filename is interpreted; the file is not opened.
    Returns:
        A ``SceneInfo`` record for a supported HH/HV raster, or ``None`` when
        the filename does not meet the expected processed NISAR convention.
    Raises:
        ValueError: A matching filename has impossible timestamps, a reversed
            acquisition interval, or conflicting source/processed product types.
    """
    processed_match = PROCESSED_PATTERN.fullmatch(path.name)  # Require a complete final-raster name.

    if processed_match is None:  # Identify unrelated files and temporary outputs.
        return None  # Instruct discovery to ignore the unsupported entry.

    processed = processed_match.groupdict()  # Collect named outer filename fields.
    source_match = SOURCE_PATTERN.fullmatch(processed["scene"])  # Validate embedded NISAR identity fields.

    if source_match is None:  # Avoid guessing Track or Frame from a partial name.
        return None  # Ignore this filename safely.

    source = source_match.groupdict()  # Collect named source-product fields.
    acquisition_start = datetime.strptime(source["acquisition_start"], "%Y%m%dT%H%M%S")  # Validate start time.
    acquisition_end = datetime.strptime(source["acquisition_end"], "%Y%m%dT%H%M%S")  # Validate end time.
    export_time = datetime.strptime(processed["export_stamp"], "%Y%m%d_%H%M%S")  # Validate export time.

    if acquisition_end < acquisition_start:  # Reject a logically impossible source interval.
        raise ValueError(f"Acquisition end precedes start in {path.name}")  # Identify the malformed raster name.

    source_product = source["source_product"].upper()  # Normalize source product type for comparison.
    product = processed["product"].upper()  # Normalize suffix product type for comparison.

    if source_product != product:  # Require one internally consistent product identity.
        raise ValueError(f"Source and processed product types differ in {path.name}")  # Refuse ambiguous grouping.

    return SceneInfo(  # Build one immutable parsed scene record.
        path=path,  # Store the source path.
        scene=processed["scene"],  # Store the original source-product text.
        level=source["level"].upper(),  # Normalize level token.
        mode=source["mode"].upper(),  # Normalize mode token.
        source_product=source_product,  # Store validated source family.
        product=product,  # Store validated processed family.
        frequency=processed["frequency"],  # Preserve frequency spelling for output naming.
        polarization=processed["pol"].upper(),  # Normalize HV/HH channel identifier.
        track=source["track"],  # Preserve Track zero padding.
        frame=source["frame"],  # Preserve Frame zero padding.
        direction=source["direction"].upper(),  # Normalize orbit direction.
        acquisition_start=acquisition_start,  # Store temporal ordering timestamp.
        acquisition_end=acquisition_end,  # Store interval end for metadata.
        export_time=export_time,  # Store duplicate-resolution timestamp.
    )  # Finish the parsed scene record.


def group_key(info: SceneInfo) -> tuple[str, ...]:
    """Return the non-temporal metadata that must match across input channels.

    Args:
        info: Parsed input-scene identity.
    Returns:
        A hashable, case-normalized match key excluding polarization and date.
    """
    return (  # Build the deterministic dictionary key.
        info.level,  # Match product processing level.
        info.mode,  # Match coverage or processing mode.
        info.source_product,  # Match embedded product type.
        info.product,  # Match processed product type.
        info.frequency.lower(),  # Match frequency without case sensitivity.
        info.track,  # Match satellite Track.
        info.frame,  # Match Frame.
        info.direction,  # Match orbit direction.
    )  # Finish the compatibility key.


def choose_scene(candidates: list[SceneInfo], acquisition_date: date, polarization: str) -> SceneInfo:
    """Choose the newest export for exactly one date and polarization.

    Args:
        candidates: Files sharing one compatibility group, date, and polarization.
        acquisition_date: Date used in any ambiguity diagnostic.
        polarization: Channel polarization label used in diagnostics.
    Returns:
        The file with the latest processed-raster export timestamp.
    Raises:
        ValueError: More than one physical acquisition exists on the date.
    """
    acquisition_times = {item.acquisition_start for item in candidates}  # Identify distinct passes on this calendar date.

    if len(acquisition_times) != 1:  # Prevent a date option from silently choosing a pass.
        names = ", ".join(item.path.name for item in sorted(candidates, key=lambda item: item.path.name))  # Gather evidence.
        raise ValueError(f"Multiple {polarization} acquisitions exist on {acquisition_date:%Y%m%d}: {names}")  # Reject ambiguity.

    selected = max(candidates, key=lambda item: (item.export_time, item.path.name))  # Prefer the newest export result.

    if len(candidates) > 1:  # Make duplicate resolution visible in the log.
        LOG.warning(  # Explain the selected export and ignored alternatives.
            "Using newest %s export for %s Track %s Frame %s: %s (%d older export(s) ignored)",  # Define message.
            polarization,  # State which polarization was resolved.
            acquisition_date.isoformat(),  # State the source date.
            selected.track,  # State Track.
            selected.frame,  # State Frame.
            selected.path.name,  # State selected file.
            len(candidates) - 1,  # State ignored duplicate count.
        )  # Complete the duplicate-resolution log entry.

    return selected  # Return the one safe raster for the channel.


def discover_composites(
    input_dir: Path,
    low_date: date | None = None,
    high_date: date | None = None,
) -> tuple[list[Composite], int]:
    """Discover compatible HV-low and HH-high temporal RGB plans.

    Args:
        input_dir: Folder containing final processed HH and HV GeoTIFFs.
        low_date: Optional known low-flood date that must contain HV.
        high_date: Optional known high-flood date that must contain HH.
    Returns:
        ``(composites, incomplete_count)``.  Without date options, the earliest
        HV date and latest later HH date are selected for each group.  With date
        options, each group must contain HV on the low date and HH on the high
        date.  Incomplete groups are logged and counted.
    Raises:
        ValueError: Only one date is supplied, dates are equal/reversed,
            recognised names are invalid, or same-day passes are ambiguous.
        OSError: The input directory cannot be enumerated.
    """
    if (low_date is None) != (high_date is None):  # Require the two override dates as one event definition.
        raise ValueError("--low-date and --high-date must be supplied together")  # Reject half-defined selection.

    if low_date is not None and high_date is not None and low_date >= high_date:  # Preserve chronological roles.
        raise ValueError("--low-date must be earlier than --high-date")  # Reject same/reversed dates.

    groups: dict[tuple[str, ...], dict[date, dict[str, list[SceneInfo]]]] = {}  # Index group, date, polarization, candidates.

    for path in sorted(input_dir.iterdir()):  # Visit folder entries in deterministic filename order.
        if not path.is_file():  # Exclude folders such as prior RGB output directories.
            continue  # Move to the next entry.

        info = filename_info(path)  # Parse one candidate filename without opening raster pixels.

        if info is None or info.polarization not in {"HV", "HH"}:  # Keep only supported channel sources.
            continue  # Ignore unknown files and non-channel products.

        by_date = groups.setdefault(group_key(info), {})  # Locate the compatible metadata group.
        by_polarization = by_date.setdefault(info.acquisition_start.date(), {})  # Locate the acquisition date.
        by_polarization.setdefault(info.polarization, []).append(info)  # Retain this polarization candidate.

    composites: list[Composite] = []  # Collect complete output plans.
    incomplete = 0  # Count groups missing a safe HV-low/HH-high pair.

    for key, by_date in sorted(groups.items()):  # Resolve every group independently.
        level, mode, source_product, product, _, track, frame, direction = key  # Extract readable identity values.
        hv_dates = sorted(item_date for item_date, channels in by_date.items() if "HV" in channels)  # List HV availability.
        hh_dates = sorted(item_date for item_date, channels in by_date.items() if "HH" in channels)  # List HH availability.

        if low_date is None or high_date is None:  # Use documented automatic temporal selection.
            selected_low_date = hv_dates[0] if hv_dates else None  # Choose earliest HV for red/blue.
            later_hh_dates = [item_date for item_date in hh_dates if selected_low_date is not None and item_date > selected_low_date]  # Enforce order.
            selected_high_date = later_hh_dates[-1] if later_hh_dates else None  # Choose latest later HH for green.
        else:  # Use the user-selected event dates exactly.
            selected_low_date = low_date  # Require HV on low date.
            selected_high_date = high_date  # Require HH on high date.

        low_available = selected_low_date is not None and "HV" in by_date.get(selected_low_date, {})  # Confirm red/blue source.
        high_available = selected_high_date is not None and "HH" in by_date.get(selected_high_date, {})  # Confirm green source.

        if not low_available or not high_available:  # Skip only this incomplete geometry group.
            incomplete += 1  # Include the group in final batch status.
            LOG.warning(  # Explain required and available channel/date combinations.
                "Skipping Track %s Frame %s %s: need HV low date and later HH high date; HV=%s; HH=%s",  # Define message.
                track,  # Identify Track.
                frame,  # Identify Frame.
                direction,  # Identify orbit direction.
                ", ".join(item.strftime("%Y%m%d") for item in hv_dates) or "none",  # Show HV dates.
                ", ".join(item.strftime("%Y%m%d") for item in hh_dates) or "none",  # Show HH dates.
            )  # Complete missing-channel log entry.
            continue  # Do not create a partial composite.

        low_hv = choose_scene(by_date[selected_low_date]["HV"], selected_low_date, "HV")  # Resolve low red/blue export.
        high_hh = choose_scene(by_date[selected_high_date]["HH"], selected_high_date, "HH")  # Resolve high green export.

        name = (  # Create a readable output stem that captures geometry and dates.
            f"NISAR_Track{track}_Frame{frame}_{direction}_{level}_{mode}_{source_product}_"  # Preserve group identity.
            f"{product}_{low_hv.frequency}_HV_HH_MultiTemporal_RGB_{selected_low_date:%Y%m%d}_{selected_high_date:%Y%m%d}"  # Preserve mapping.
        )  # Finish output-stem construction.

        composites.append(Composite(name=name, low_hv=low_hv, high_hh=high_hh))  # Save the validated plan.

    return composites, incomplete  # Return all plans and skipped-group count.


def validate_temporal_pair(low_hv_path: Path, high_hh_path: Path) -> tuple[SceneInfo, SceneInfo]:
    """Verify HV-low and HH-high filenames are compatible before raster I/O.

    Args:
        low_hv_path: Low-flood HV GeoTIFF for output red and blue.
        high_hh_path: High-flood HH GeoTIFF for output green.
    Returns:
        Parsed ``(low_hv, high_hh)`` source records.
    Raises:
        ValueError: Names are unsupported, polarizations are incorrect, match
            metadata differs, or acquisition dates are not chronologically ordered.
    """
    low_hv = filename_info(low_hv_path)  # Parse the proposed red/blue source.
    high_hh = filename_info(high_hh_path)  # Parse the proposed green source.

    if low_hv is None or high_hh is None:  # Require complete trustworthy NISAR names.
        raise ValueError("Both inputs must follow the final NISAR processed GeoTIFF naming convention")  # Reject guesses.

    if low_hv.polarization != "HV" or high_hh.polarization != "HH":  # Enforce requested channel mapping.
        raise ValueError("Red/blue input must be HV and green input must be HH")  # Explain exact requirement.

    if group_key(low_hv) != group_key(high_hh):  # Compare all non-temporal identity fields.
        raise ValueError("Inputs must share Track, Frame, direction, level, mode, product, and frequency")  # Prevent mixing grids.

    if low_hv.acquisition_start.date() >= high_hh.acquisition_start.date():  # Preserve low/high temporal roles.
        raise ValueError("HV low-flood acquisition date must be earlier than HH high-flood acquisition date")  # Reject reversal.

    return low_hv, high_hh  # Supply verified source metadata to the writer.


def stack_temporal_pair(low_hv_path: Path, high_hh_path: Path, output_path: Path) -> None:
    """Write one aligned float32 RGB GeoTIFF with HV, HH, HV channel mapping.

    Args:
        low_hv_path: Single-band low/non-flood HV dB raster for red and blue.
        high_hh_path: Single-band high-flood HH dB raster for green.
        output_path: Destination GeoTIFF path; its parent must already exist.
    Returns:
        ``None`` after atomically publishing the completed output.
    Raises:
        ValueError: Filename identity, grid, CRS, band count, or output safety
            requirements are not met.
        OSError or rasterio.errors.RasterioError: Raster reading or writing fails.
    Notes:
        A pixel is valid only when both source pixels are valid and finite.  The
        final path is not replaced until pixels, mask, tags, and overviews exist.
    """
    low_hv_info, high_hh_info = validate_temporal_pair(low_hv_path, high_hh_path)  # Validate direct calls too.
    resolved_output = output_path.resolve()  # Normalize output before comparing it to source paths.

    if resolved_output in {low_hv_path.resolve(), high_hh_path.resolve()}:  # Protect source rasters from truncation.
        raise ValueError("Output must not replace either input raster")  # Stop before opening any writer.

    temporary = output_path.with_name(f".{uuid.uuid4().hex}.part.tif")  # Stage output next to its final destination.

    try:  # Ensure only this invocation's temporary file is removed after failure.
        with rasterio.open(low_hv_path) as low_hv, rasterio.open(high_hh_path) as high_hh:  # Open and close source datasets.
            if low_hv.count != 1 or high_hh.count != 1:  # Require original single-band dB products.
                raise ValueError("Expected single-band HV and HH inputs")  # Reject stacks or unexpected rasters.

            if low_hv.crs is None or high_hh.crs is None:  # Require georeferencing before spatial comparison.
                raise ValueError("Both inputs must have a coordinate reference system")  # Reject unlocated pixels.

            if low_hv.crs != high_hh.crs or low_hv.shape != high_hh.shape or low_hv.transform != high_hh.transform:  # Compare exact grids.
                raise ValueError("HV and HH grids differ; align CRS, dimensions, and transform before stacking")  # Do not resample silently.

            profile = {  # Define georeferencing, storage, and display characteristics.
                "driver": "GTiff",  # Write a GeoTIFF dataset.
                "width": low_hv.width,  # Preserve shared pixel width.
                "height": low_hv.height,  # Preserve shared pixel height.
                "count": 3,  # Allocate red, green, and blue bands.
                "dtype": "float32",  # Preserve continuous dB values.
                "crs": low_hv.crs,  # Preserve common CRS.
                "transform": low_hv.transform,  # Preserve common pixel grid.
                "nodata": OUTPUT_NODATA,  # Declare common invalid-pixel value.
                "tiled": True,  # Support efficient windowed access.
                "blockxsize": TILE_SIZE,  # Set tile width.
                "blockysize": TILE_SIZE,  # Set tile height.
                "compress": "deflate",  # Apply lossless compression.
                "predictor": 3,  # Improve float compression performance.
                "BIGTIFF": "IF_SAFER",  # Enable BigTIFF only when needed.
                "photometric": "RGB",  # Advertise the required RGB interpretation.
                "interleave": "pixel",  # Store three channel samples together per pixel.
            }  # Finish destination profile.

            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):  # Embed validity mask in the TIFF file.
                with rasterio.open(temporary, "w", **profile) as dst:  # Create and safely close staging raster.
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)  # Declare band display roles.

                    low_date_text = low_hv_info.acquisition_start.strftime("%Y-%m-%d")  # Format low date for labels.
                    high_date_text = high_hh_info.acquisition_start.strftime("%Y-%m-%d")  # Format high date for labels.
                    band_labels = (  # Define human-readable labels in raster-band order.
                        f"HV non-flood/low-flood ({low_date_text})",  # Label red channel.
                        f"HH high-flood ({high_date_text})",  # Label green channel.
                        f"HV non-flood/low-flood ({low_date_text})",  # Label blue channel.
                    )  # Finish labels.

                    for index, label in enumerate(band_labels, 1):  # Rasterio bands use one-based indexing.
                        dst.set_band_description(index, label)  # Store channel role and date.
                        dst.set_band_unit(index, "dB")  # Store measurement unit.

                    dst.update_tags(  # Embed source provenance and channel semantics.
                        SOURCE_LOW_HV=low_hv_path.name,  # Identify red/blue source exactly.
                        SOURCE_HIGH_HH=high_hh_path.name,  # Identify green source exactly.
                        TRACK=low_hv_info.track,  # Record shared Track.
                        FRAME=low_hv_info.frame,  # Record shared Frame.
                        ORBIT_DIRECTION=low_hv_info.direction,  # Record shared orbit direction.
                        PRODUCT_TYPE=low_hv_info.product,  # Record shared processed product type.
                        FREQUENCY=low_hv_info.frequency,  # Record shared radar frequency.
                        LOW_HV_ACQUISITION_START=low_hv_info.acquisition_start.isoformat(),  # Record red/blue start.
                        LOW_HV_ACQUISITION_END=low_hv_info.acquisition_end.isoformat(),  # Record red/blue end.
                        HIGH_HH_ACQUISITION_START=high_hh_info.acquisition_start.isoformat(),  # Record green start.
                        HIGH_HH_ACQUISITION_END=high_hh_info.acquisition_end.isoformat(),  # Record green end.
                        LOW_HV_EXPORT_TIMESTAMP=low_hv_info.export_time.isoformat(),  # Record red/blue export.
                        HIGH_HH_EXPORT_TIMESTAMP=high_hh_info.export_time.isoformat(),  # Record green export.
                        PAIRING_RULE="Same Track, Frame, direction, level, mode, product, and frequency; HV low date before HH high date",  # Explain pair.
                        BAND_MAPPING="R=low/non-flood HV; G=high-flood HH; B=low/non-flood HV",  # Explain channels.
                        VALIDITY="All bands valid only where both HV and HH source pixels are valid and finite",  # Explain mask.
                    )  # Finish metadata update.

                    tile_rows = (low_hv.height + TILE_SIZE - 1) // TILE_SIZE  # Round up tile-row count.
                    tile_columns = (low_hv.width + TILE_SIZE - 1) // TILE_SIZE  # Round up tile-column count.
                    total = tile_rows * tile_columns  # Count tiles for progress messages.

                    for number, (_, window) in enumerate(dst.block_windows(1), 1):  # Process every output tile once.
                        low_hv_data = low_hv.read(1, window=window, masked=True, out_dtype="float32")  # Read HV data/mask.
                        high_hh_data = high_hh.read(1, window=window, masked=True, out_dtype="float32")  # Read HH data/mask.
                        valid = ~np.ma.getmaskarray(low_hv_data) & ~np.ma.getmaskarray(high_hh_data)  # Require both masks valid.
                        valid &= np.isfinite(low_hv_data.data) & np.isfinite(high_hh_data.data)  # Exclude NaN/infinity.
                        data = np.stack((low_hv_data.data, high_hh_data.data, low_hv_data.data)).astype(np.float32, copy=False)  # Map HV/HH/HV.
                        data[:, ~valid] = OUTPUT_NODATA  # Fill all invalid channel pixels consistently.
                        dst.write(data, window=window)  # Write completed RGB tile.
                        dst.write_mask(valid.astype(np.uint8) * 255, window=window)  # Write common valid-pixel mask.

                        if number == total or number % max(1, total // 10) == 0:  # Log roughly every ten percent.
                            LOG.info("Stacking: %.0f%% (%d/%d tiles)", 100 * number / total, number, total)  # Report progress.

                    factors = [factor for factor in (2, 4, 8, 16, 32) if min(low_hv.width, low_hv.height) // factor >= 1]  # Fit pyramids.

                    if factors:  # Skip invalid overview construction for tiny rasters.
                        dst.build_overviews(factors, Resampling.average)  # Build continuous-data display pyramids.
                        dst.update_tags(ns="rio_overview", resampling="average")  # Record overview method.

        os.replace(temporary, output_path)  # Atomically publish only the completed result.

    finally:  # Run after either success or a writing exception.
        if temporary.exists():  # Detect an incomplete staged output.
            temporary.unlink()  # Remove only the unique temporary file for this call.


def main() -> int:
    """Run the documented command-line batch workflow.

    Returns:
        ``0`` when every complete group is written or already exists; ``1`` on
        discovery failure, no valid pairs, incomplete groups, or write failures.
    Side effects:
        Creates the output directory and writes requested GeoTIFF composites.
    """
    parser = argparse.ArgumentParser(  # Configure detailed command-line help.
        description=__doc__,  # Reuse the module guide as command description.
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve intended help formatting.
    )  # Complete argument parser construction.

    parser.add_argument(  # Add the processed-raster source folder option.
        "--input-dir",  # Expose a readable long option.
        type=Path,  # Convert supplied text into a platform-safe path.
        default=DEFAULT_INPUT_DIRECTORY,  # Use normal NISAR processed output by default.
        help="Folder containing final processed HV and HH GeoTIFFs",  # Explain required channel inputs.
    )  # Finish source-folder option.

    parser.add_argument(  # Add optional destination override.
        "--output-dir",  # Expose a readable long option.
        type=Path,  # Convert supplied text into a platform-safe path.
        help="Default: INPUT_DIR/Multi_Temporal_HV_HH_RGB",  # Explain automatic destination.
    )  # Finish destination-folder option.

    parser.add_argument(  # Add known low-flood date selection.
        "--low-date",  # Use wording tied to red and blue mapping.
        type=parse_date_argument,  # Validate compact date format immediately.
        help="Low/non-flood HV date (YYYYMMDD); requires --high-date",  # Explain paired HV requirement.
    )  # Finish low-date option.

    parser.add_argument(  # Add known high-flood date selection.
        "--high-date",  # Use wording tied to green mapping.
        type=parse_date_argument,  # Validate compact date format immediately.
        help="High-flood HH date (YYYYMMDD); requires --low-date",  # Explain paired HH requirement.
    )  # Finish high-date option.

    parser.add_argument(  # Add explicit output-replacement choice.
        "--overwrite",  # Expose a readable long switch.
        action="store_true",  # Keep existing composites unless requested otherwise.
        help="Replace existing composites; otherwise skip them",  # Explain safe default.
    )  # Finish overwrite option.

    args = parser.parse_args()  # Parse and validate supplied options.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")  # Configure console logs.

    if not args.input_dir.is_dir():  # Verify source folder before making output directories.
        LOG.error("Input directory does not exist: %s", args.input_dir)  # Identify missing source.
        return 1  # Signal command failure.

    try:  # Convert predictable discovery issues into concise command output.
        composites, incomplete = discover_composites(args.input_dir, args.low_date, args.high_date)  # Build safe plans.
    except (ValueError, OSError) as exc:  # Catch date, name, ambiguity, and enumeration failures.
        LOG.error("Composite discovery failed: %s", exc)  # Report processing blocker.
        return 1  # Stop before creating output.

    if not composites:  # Detect when no complete compatible pair was found.
        LOG.error("No compatible HV low-date and later HH high-date scene pairs were found")  # Explain no-output outcome.
        return 1  # Signal no work was completed.

    output_dir = args.output_dir or args.input_dir / DEFAULT_OUTPUT_SUBDIRECTORY  # Select requested/default output folder.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create destination and missing parents.
    LOG.info("Found %d HV/HH multi-temporal pair(s)", len(composites))  # Report planned output count.

    created = skipped = failed = 0  # Initialize batch outcome counters.

    for composite in composites:  # Process each independent Track/Frame plan.
        output_path = output_dir / f"{composite.name}.tif"  # Form final GeoTIFF destination.

        if output_path.exists() and not args.overwrite:  # Preserve completed result unless replacement was requested.
            LOG.info("Already exists, skipping: %s", output_path)  # Report retained output.
            skipped += 1  # Count skip.
            continue  # Continue with next plan.

        try:  # Let later independent groups proceed after one raster failure.
            LOG.info("Low/non-flood HV for red/blue: %s", composite.low_hv.path.name)  # Report source selection.
            LOG.info("High-flood HH for green: %s", composite.high_hh.path.name)  # Report source selection.
            stack_temporal_pair(composite.low_hv.path, composite.high_hh.path, output_path)  # Write composite.
            created += 1  # Count successful output.
            LOG.info("Created: %s", output_path)  # Report output location.
        except Exception:  # Catch unexpected Rasterio and filesystem errors per group.
            failed += 1  # Count failed output.
            LOG.exception("Failed composite: %s", composite.name)  # Include traceback for diagnosis.

    LOG.info("Done: %d created, %d existing, %d incomplete, %d failed", created, skipped, incomplete, failed)  # Summarize batch.
    return 1 if failed or incomplete else 0  # Signal partial failure when any group could not be handled.


if __name__ == "__main__":  # Run the batch only when this module is executed directly.
    raise SystemExit(main())  # Return workflow status to the shell.
