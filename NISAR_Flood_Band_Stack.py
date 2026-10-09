#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a three-band RGB GeoTIFF from final NISAR HH and HV dB rasters.

Purpose
-------
Read final outputs created by NISAR_Process.py in GeoTIFF_Processed and stack:
    Band 1 / Red   = HV (dB)
    Band 2 / Green = HH (dB)
    Band 3 / Blue  = HH (dB) - HV (dB), i.e. the HH/HV ratio in dB

Pairing
-------
HH and HV must have the SAME full source product name, product type, and
frequency. Matching the full source name preserves any acquisition date/time
identifiers in that name; different source products are never combined, even
when acquired on the same day.
The export timestamp (YYYYMMDD_HHMMSS) is NOT required to match when a source
has exactly one HH file and one HV file: HH and HV exported by separate runs of
NISAR_Process.py are paired, and both timestamps are logged and recorded in the
output metadata. When a source has several HH or HV files (multiple runs of the
same channel), the export timestamp must match exactly, so runs are never mixed
by guesswork. The output name uses the later of the two timestamps.
Dates come from filenames produced by NISAR_Process.py. Filesystem creation and
modification dates are not used because copying a file can change those dates.
If a required channel is missing for a group, the group is reported and skipped.

Input filename convention
-------------------------
<SOURCE>_<GSLC|GCOV>_<FREQUENCY>_<HH|HV>_Processed_dB_<YYYYMMDD_HHMMSS>.tif
Example matching pair:
    sceneA_GCOV_frequencyA_HH_Processed_dB_20260904_120000.tif
    sceneA_GCOV_frequencyA_HV_Processed_dB_20260904_120000.tif
An HV file ending in 20260905_120000.tif cannot match the example HH file.
Temporary .part.tif and .native.part.tif files are excluded automatically.

Processing and outputs
----------------------
The inputs are already in decibels, so blue (the HH/HV ratio) is a difference of dB values.
All three output bands retain float32 values without a display stretch.
CRS, transform, dimensions, and pixel spacing are preserved. Different grids
are rejected; this script does not resample either input. A pixel is valid in
all three bands only when both source pixels are valid and finite.
Tiled reading/writing bounds memory usage for large scenes. Outputs use lossless
compression, an internal validity mask, and nearest-resampled display overviews.
Failed source reads are retried once using a newly opened dataset. Persistent
read failures reject the pair and remove its temporary stack; subsequent pairs
continue. Unreadable pixels are never silently substituted with valid values.

Default input:  GeoTIFF_Processed
Default output: GeoTIFF_Processed\\Band_Stacked
Existing stacks are skipped unless --overwrite is supplied.

Dependencies and usage
----------------------
Use the Python environment containing numpy and rasterio that runs the original
NISAR_Process.py. This separate script does not import or modify that script.
    python NISAR_Flood_Band_Stack.py
    python NISAR_Flood_Band_Stack.py --input-dir "GeoTIFF_Processed"
    python NISAR_Flood_Band_Stack.py --output-dir "NISAR\\RGB"
    python NISAR_Flood_Band_Stack.py --overwrite
In QGIS choose Multiband color, Red=1, Green=2, Blue=3, with a per-band stretch.

Function guide
--------------
filename_info: Parse and validate the identifiers and date in an input filename.
discover_pairs: Find complete HH/HV pairs with identical source identity.
validate_pair: Recheck source/channel identity before opening a requested pair.
stack_pair: Validate spatial grids and write the three-band output tile by tile.
main: Parse command-line options, process the batch, and report its outcome.
"""

from __future__ import annotations  # Defer evaluation of type annotations.

import argparse  # Read input/output folder and overwrite options from the command line.
import logging  # Report progress, missing pairs, and processing errors.
import os  # Promote a completed temporary raster to its final filename.
from datetime import datetime  # Validate calendar dates and times from filenames.
from pathlib import Path  # Handle filesystem paths without manual path separators.
import re  # Recognize the final GeoTIFF filename convention.
import uuid  # Give each temporary raster a unique filename.

import numpy as np  # Calculate the difference band and pixel validity arrays.
import rasterio  # Read and write georeferenced raster datasets.
from rasterio.enums import ColorInterp, Resampling  # Label RGB channels and build overviews.
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

DEFAULT_INPUT_DIRECTORY = Path(r"GeoTIFF_Processed")  # Locate final rasters.
TILE_SIZE = 512  # Process at most 512 by 512 pixels in each tile.
OUTPUT_NODATA = -9999.0  # Represent invalid pixels consistently across all output bands.
INPUT_PATTERN = re.compile(  # Describe the complete expected input filename.
    r"^(?P<scene>.+)_(?P<product>GSLC|GCOV)_(?P<frequency>frequency[^_]+)_"  # Capture source and grid.
    r"(?P<pol>HH|HV)_Processed_dB_(?P<stamp>\d{8}_\d{6})\.tiff?$",  # Capture channel and export time.
    re.IGNORECASE,  # Accept uppercase or lowercase filename extensions and labels.
)  # Finish compiling the filename pattern.
IDENTITY_FIELDS = ("scene", "product", "frequency")  # Required pair matches (export time may differ).
LOG = logging.getLogger("NISAR_Band_Stack")  # Use one logger for all script functions.


class SourceReadError(RasterioIOError):
    """A source window remained unreadable after reopening the dataset."""


def read_source_window(source: rasterio.io.DatasetReader, window: Window) -> np.ma.MaskedArray:
    """Read a source tile, retrying once with a fresh handle after an I/O error.

    Reopening permits recovery from a transient read failure without suppressing
    TIFF decoding errors or accepting a partially decoded array. A persistent
    failure includes the source path and pixel window in its diagnostic.
    """
    location = (
        f"column={int(window.col_off)}, row={int(window.row_off)}, "
        f"width={int(window.width)}, height={int(window.height)}"
    )
    with rasterio.Env(GTIFF_IGNORE_READ_ERRORS=False):
        try:
            return source.read(1, window=window, masked=True, out_dtype="float32")
        except RasterioIOError as exc:
            LOG.warning("Source read failed: %s (%s): %s. Reopening and retrying once.",
                        source.name, location, exc)

        try:
            with rasterio.open(source.name, sharing=False) as reopened:
                data = reopened.read(1, window=window, masked=True, out_dtype="float32")
        except RasterioIOError as exc:
            raise SourceReadError(
                f"Source remains unreadable after retry: {source.name} ({location}). "
                "Check the source file and storage. If the failure persists, regenerate "
                "HH and HV together from the original product with NISAR_Process.py "
                "so their export timestamps match. Move the previous HH/HV exports "
                "out of GeoTIFF_Processed first, because existing exports are skipped."
            ) from exc

    LOG.warning("Source read recovered after reopening: %s (%s)", source.name, location)
    return data


def filename_info(path: Path) -> dict[str, str] | None:  # Define the filename parser.
    """Extract source/channel identity and a validated export date from a filename.

    Args:
        path: Candidate GeoTIFF path; this function examines only its filename.
    Returns:
        A dictionary with scene, product, frequency, pol, stamp, and export_date,
        or None when the filename is not a supported final processed raster.
    Raises:
        ValueError: A matching filename contains an impossible date or time.
    """
    match = INPUT_PATTERN.fullmatch(path.name)  # Match the whole name, excluding temporary files.

    if match is None:  # Detect unrelated filenames.
        return None  # Tell discovery to ignore this file.

    info = match.groupdict()  # Convert named regex fields into a dictionary.
    exported = datetime.strptime(info["stamp"], "%Y%m%d_%H%M%S")  # Validate the timestamp.

    info["export_date"] = exported.strftime("%Y-%m-%d")  # Record the export calendar date explicitly.
    info["pol"] = info["pol"].upper()  # Normalize HH/HV channel labels.
    info["product"] = info["product"].upper()  # Normalize GSLC/GCOV product labels.

    return info  # Return all validated filename identifiers.


def discover_pairs(input_dir: Path) -> tuple[list[tuple[str, Path, Path]], int]:  # Define pair discovery.
    """Find complete HH/HV pairs with identical source identity in one folder.

    A source with one HH and one HV file is paired even when their export
    timestamps differ. A source with several HH or HV files is paired only by
    identical export timestamp.

    Args:
        input_dir: Directory containing final GeoTIFFs from NISAR_Process.py.
    Returns:
        (pairs, incomplete_count). Each pair contains an output filename stem,
        its HH path, and its HV path. Every complete pair is included.
    Raises:
        ValueError: A recognized filename has an invalid date, or two files
            claim the same channel and timestamp (for example .tif/.tiff).
        OSError: The input directory cannot be read.
    """
    # source identity -> channel -> export timestamp -> path
    groups: dict[tuple[str, ...], dict[str, dict[str, Path]]] = {}

    for path in sorted(input_dir.iterdir()):  # Inspect files in deterministic filename order.
        if not path.is_file():  # Exclude folders, including the output subfolder.
            continue  # Move to the next directory entry.

        info = filename_info(path)  # Extract and validate this candidate's identifiers.
        if info is None:  # Detect unrelated files and unfinished raster outputs.
            continue  # Ignore entries that cannot be safely paired.

        identity = tuple(info[field] for field in IDENTITY_FIELDS)  # Include source, product, and frequency.
        runs = groups.setdefault(identity, {}).setdefault(info["pol"], {})  # Files of this channel by export time.

        if info["stamp"] in runs:  # Detect multiple candidates for a single channel and run.
            raise ValueError(f"Ambiguous {info['pol']} pair: {runs[info['stamp']]} and {path}")  # Refuse to guess.

        runs[info["stamp"]] = path  # Register this channel's export.

    pairs: list[tuple[str, Path, Path]] = []  # Collect only complete and unambiguous pairs.
    incomplete = 0  # Count exports lacking a required counterpart.

    for identity, channels in sorted(groups.items()):  # Examine each source independently.
        scene, product, frequency = identity  # Unpack the shared identifiers.
        label = f"{scene}_{product}_{frequency}"  # Describe this source in log messages.
        hh_runs = channels.get("HH", {})  # HH exports by timestamp.
        hv_runs = channels.get("HV", {})  # HV exports by timestamp.

        if len(hh_runs) == 1 and len(hv_runs) == 1:  # Unambiguous: pair regardless of export time.
            matches = [(next(iter(hh_runs)), next(iter(hv_runs)))]
        else:  # Missing or multiple runs: require identical export timestamps.
            matches = [(stamp, stamp) for stamp in sorted(hh_runs.keys() & hv_runs.keys())]
            unmatched = [("HV", stamp) for stamp in sorted(hh_runs.keys() - hv_runs.keys())]  # HH without HV.
            unmatched += [("HH", stamp) for stamp in sorted(hv_runs.keys() - hh_runs.keys())]  # HV without HH.

            for missing, stamp in unmatched:  # Report each export lacking its counterpart.
                incomplete += 1  # Count this unprocessed export.
                LOG.warning("Skipping %s (export %s): missing %s", label, stamp, missing)  # Explain why.

        for hh_stamp, hv_stamp in matches:  # Register each accepted pair.
            stamp = max(hh_stamp, hv_stamp)  # Name the output after the later export.
            name = f"{label}_Flood_Bandstack_{stamp}"  # Name the output.

            if hh_stamp != hv_stamp:  # Make the differing run times visible.
                LOG.warning("HH (%s) and HV (%s) export timestamps differ for %s; pairing them "
                            "because they are the only HH/HV files for this source",
                            hh_stamp, hv_stamp, label)

            pairs.append((name, hh_runs[hh_stamp], hv_runs[hv_stamp]))  # Preserve HH-first channel order.

    return pairs, incomplete  # Return the complete pair list and skipped-export count.


def validate_pair(hh_path: Path, hv_path: Path) -> dict[str, str]:  # Define the direct-call pairing guard.
    """Verify channel order and matching source identifiers before stacking.

    Args:
        hh_path: Final processed HH GeoTIFF path.
        hv_path: Final processed HV GeoTIFF path.
    Returns:
        Validated HH filename metadata, also describing the pair's shared identity.
    Raises:
        ValueError: A filename/date is invalid, channels are reversed, or any
            source, product, or frequency differs. Export timestamps may differ.
    """
    hh_info = filename_info(hh_path)  # Parse the proposed red-band input.
    hv_info = filename_info(hv_path)  # Parse the proposed green-band input.

    if hh_info is None or hv_info is None:  # Require recognizable final raster filenames.
        raise ValueError("Both inputs must follow the NISAR_Process.py final GeoTIFF naming convention")  # Reject unknown names.

    if hh_info["pol"] != "HH" or hv_info["pol"] != "HV":  # Verify inputs have the intended channels.
        raise ValueError("Expected HH as the first input and HV as the second input")  # Reject swapped channels.

    different = [field for field in IDENTITY_FIELDS if hh_info[field] != hv_info[field]]  # Compare pairing identifiers.

    if different:  # Reject cross-source input combinations.
        raise ValueError("HH/HV must share source, product, and frequency; differing fields: " + ", ".join(different))  # Explain mismatch.

    return hh_info  # Supply the validated pair metadata to the writer.


def stack_pair(hh_path: Path, hv_path: Path, output_path: Path) -> None:  # Define the raster stacking function.
    """Write an aligned HH/HV pair as a georeferenced RGB float32 stack.

    Args:
        hh_path: Single-band HH raster in dB, opened read-only.
        hv_path: Single-band HV raster in dB, opened read-only.
        output_path: Destination GeoTIFF; its parent directory must exist.
    Returns:
        None. Writes a complete raster with R=HV, G=HH, B=HH/HV (HH-HV in dB) and metadata.
    Raises:
        ValueError: Pair identifiers, channels, band counts, or spatial grids
            are incompatible, or the output path would replace an input raster.
        OSError or rasterio.errors.RasterioError: Raster reading/writing fails.
    Notes:
        Shared masking excludes NoData, NaN, and infinity in either source.
        The final filename is published only after writing and overviews finish.
        This function replaces an existing output; main enforces --overwrite.
    """
    info = validate_pair(hh_path, hv_path)  # Recheck source identity even for direct function calls.
    hv_info = filename_info(hv_path)  # Parse HV's export time, which may differ from HH's.

    if output_path.resolve() in {hh_path.resolve(), hv_path.resolve()}:  # Protect both input files.
        raise ValueError("Output must not replace an input raster")  # Stop before opening a writer.

    temporary = output_path.with_name(f".{uuid.uuid4().hex}.part.tif")  # Allocate a unique staging path.

    try:  # Ensure staging-file cleanup on success or failure.
        with rasterio.open(hh_path) as hh, rasterio.open(hv_path) as hv:  # Open and automatically close both sources.
            if hh.count != 1 or hv.count != 1:  # Require the original script's single-band outputs.
                raise ValueError("Expected single-band HH and HV inputs")  # Reject already stacked files.

            if hh.crs is None or hv.crs is None:  # Require known geographic coordinate systems.
                raise ValueError("Both inputs must have a coordinate reference system")  # Stop on missing georeferencing.

            if hh.crs != hv.crs or hh.shape != hv.shape or hh.transform != hv.transform:  # Require identical pixel grids.
                raise ValueError("HH/HV grids differ; align CRS, dimensions, and transform before stacking")  # Avoid spatially wrong subtraction.

            profile = {  # Define the output's storage and georeferencing settings.
                "driver": "GTiff",  # Write a GeoTIFF dataset.
                "width": hh.width,  # Preserve the input column count.
                "height": hh.height,  # Preserve the input row count.
                "count": 3,  # Allocate red, green, and blue bands.
                "dtype": "float32",  # Retain fractional and negative dB values.
                "crs": hh.crs,  # Preserve the common coordinate reference system.
                "transform": hh.transform,  # Preserve origin, pixel spacing, and orientation.
                "nodata": OUTPUT_NODATA,  # Declare the invalid-pixel sentinel.
                "tiled": True,  # Organize the output into independently readable tiles.
                "blockxsize": TILE_SIZE,  # Set tile width in pixels.
                "blockysize": TILE_SIZE,  # Set tile height in pixels.
                "compress": "deflate",  # Compress data without changing pixel values.
                "predictor": 3,  # Improve compression of floating-point values.
                "BIGTIFF": "IF_SAFER",  # Allow files larger than the classic TIFF size limit.
                "photometric": "RGB",  # Identify the three bands as color channels.
                "interleave": "pixel",  # Store the channels together for efficient RGB viewing.
            }  # Finish the output profile.

            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):  # Embed the mask so it follows the TIFF rename.
                with rasterio.open(temporary, "w", **profile) as dst:  # Create and automatically close the staging raster.
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)  # Set R/G/B band roles.

                    for index, label in enumerate(("HV (dB)", "HH (dB)", "HH/HV (dB)"), 1):  # Number bands from one.
                        dst.set_band_description(index, label)  # Attach a readable band label.
                        dst.set_band_unit(index, "dB")  # Record the numeric unit for this band.

                    dst.update_tags(  # Record provenance, pairing evidence, and calculation details.
                        SOURCE_HH=hh_path.name,  # Identify the HH source file (green channel).
                        SOURCE_HV=hv_path.name,  # Identify the HV source file (red channel).
                        SOURCE_PRODUCT=info["scene"],  # Preserve the source product and any acquisition identifiers.
                        SOURCE_HH_EXPORT_TIMESTAMP=info["stamp"],  # Record the HH processing run timestamp.
                        SOURCE_HV_EXPORT_TIMESTAMP=hv_info["stamp"],  # Record the HV processing run timestamp.
                        PAIRING_RULE="Same source product, product type and frequency; export timestamps may differ",  # Describe matching.
                        BAND_MAPPING="R=HV; G=HH; B=HH/HV",  # Document output band order.
                        BLUE_FORMULA="HH_dB - HV_dB",  # HH/HV ratio expressed as a dB difference.
                        VALIDITY="All bands valid only where both source bands are valid",  # Describe the shared footprint.
                    )  # Finish dataset metadata.

                    tile_rows = (hh.height + TILE_SIZE - 1) // TILE_SIZE  # Round up the number of tile rows.
                    tile_columns = (hh.width + TILE_SIZE - 1) // TILE_SIZE  # Round up the number of tile columns.
                    total = tile_rows * tile_columns  # Count tiles for progress reporting.

                    for number, (_, window) in enumerate(dst.block_windows(1), 1):  # Visit each output tile once.
                        hh_tile = read_source_window(hh, window)  # Read HH and retry transient I/O failures.
                        hv_tile = read_source_window(hv, window)  # Read matching HV pixels with the same checks.

                        valid = ~np.ma.getmaskarray(hh_tile) & ~np.ma.getmaskarray(hv_tile)  # Require both source masks to be valid.
                        valid &= np.isfinite(hh_tile.data) & np.isfinite(hv_tile.data)  # Exclude NaN and infinity in either source.

                        blue = np.full(hh_tile.shape, OUTPUT_NODATA, dtype=np.float32)  # Initialize the blue tile as invalid.
                        with np.errstate(over="ignore", invalid="ignore"):  # Handle nonfinite arithmetic through validity checks.
                            np.subtract(hh_tile.data, hv_tile.data, out=blue, where=valid)  # HH/HV ratio in dB = HH minus HV.

                        valid &= np.isfinite(blue)  # Reject any nonfinite subtraction results.
                        data = np.stack((hv_tile.data, hh_tile.data, blue))  # Arrange as R=HV, G=HH, B=HH/HV.
                        data[:, ~valid] = OUTPUT_NODATA  # Apply the same invalid footprint to all bands.

                        dst.write(data, window=window)  # Write the three completed tile bands.
                        dst.write_mask(valid.astype(np.uint8) * 255, window=window)  # Write 255 for valid and 0 for invalid pixels.

                        if number == total or number % max(1, total // 10) == 0:  # Log roughly every ten percent and at completion.
                            LOG.info("Stacking: %.0f%% (%d/%d tiles)", 100 * number / total, number, total)  # Report progress.

                    factors = [f for f in (2, 4, 8, 16, 32) if min(hh.width, hh.height) // f >= 1]  # Select valid overview sizes.

                    if factors:  # Avoid creating overviews for rasters too small to downsample.
                        dst.build_overviews(factors, Resampling.nearest)  # Build display pyramids without altering base pixels.
                        dst.update_tags(ns="rio_overview", resampling="nearest")  # Record how overview pixels were computed.

        os.replace(temporary, output_path)  # Publish the completed raster after closing all dataset handles.

    finally:  # Run cleanup even if reading, writing, or validation fails.
        if temporary.exists():  # Check whether this call left its unique staging file behind.
            temporary.unlink()  # Remove only that incomplete temporary raster.


def main() -> int:  # Define the command-line batch workflow.
    """Read options, discover valid pairs, process stacks, and summarize results.

    Args:
        None. Reads --input-dir, --output-dir, and --overwrite from the command line.
    Returns:
        0 when all discovered groups are created or already exist; 1 if input
        discovery fails, no complete pairs exist, or any group is incomplete/fails.
    Side effects:
        Creates the output directory, writes stacked GeoTIFFs, and logs progress.
        A failed pair is reported while processing continues with remaining pairs.
    """
    parser = argparse.ArgumentParser(  # Configure readable command-line help.
        description=__doc__,  # Use the module documentation to explain the workflow.
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve documentation line breaks.
    )  # Finish parser construction.

    parser.add_argument(  # Define the source-folder option.
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY,  # Use the original processing folder by default.
        help="Folder containing final HH/HV processed GeoTIFFs",  # Explain the option in --help.
    )  # Finish the input-directory option.

    parser.add_argument(  # Define an optional output-folder override.
        "--output-dir", type=Path, help="Default: INPUT_DIR/Band_Stacked",  # Otherwise use a source-folder subdirectory.
    )  # Finish the output-directory option.

    parser.add_argument(  # Define explicit replacement of existing stacked outputs.
        "--overwrite", action="store_true", help="Replace existing stacks; otherwise skip them",  # Default to keeping completed files.
    )  # Finish the overwrite option.

    args = parser.parse_args()  # Read and validate the user's command-line arguments.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")  # Configure timestamped messages.

    if not args.input_dir.is_dir():  # Check that the requested input folder exists.
        LOG.error("Input directory does not exist: %s", args.input_dir)  # Explain the missing source folder.
        return 1  # Signal failure to the calling shell.

    try:  # Handle discovery failures with a concise error message.
        pairs, incomplete = discover_pairs(args.input_dir)  # Collect exact source/date/run matches.
    except (ValueError, OSError) as exc:  # Catch ambiguous names, invalid dates, and unreadable directories.
        LOG.error("Pair discovery failed: %s", exc)  # Report the reason pairing could not proceed.
        return 1  # Stop without producing potentially mismatched stacks.

    if not pairs:  # Detect an empty folder or groups with no complete channel pair.
        LOG.error("No complete same-source HH/HV pairs found")  # Explain why nothing can be stacked.
        return 1  # Signal that no usable input pair was available.

    output_dir = args.output_dir or args.input_dir / "Band_Stacked"  # Choose the requested or default destination.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder and any missing parents.
    LOG.info("Found %d complete HH/HV pair(s)", len(pairs))  # Report the batch size.

    created = skipped = failed = 0  # Initialize output, existing-file, and failure counters.

    for name, hh_path, hv_path in pairs:  # Process each independently matched HH/HV pair.
        output_path = output_dir / f"{name}.tif"  # Construct this pair's final stack path.

        if output_path.exists() and not args.overwrite:  # Preserve existing stacks unless replacement was requested.
            LOG.info("Already exists, skipping: %s", output_path)  # Identify the retained output.
            skipped += 1  # Count this existing stack.
            continue  # Move to the next pair without rewriting this output.

        try:  # Isolate processing failures to the affected pair.
            LOG.info("HH: %s | HV: %s", hh_path.name, hv_path.name)  # Show exactly which source files are paired.
            stack_pair(hh_path, hv_path, output_path)  # Validate and write the RGB raster.
            created += 1  # Count the newly completed stack.
            LOG.info("Created: %s", output_path)  # Report the final output path.
        except SourceReadError as exc:
            failed += 1
            LOG.error("Failed pair: %s. %s", name, exc)
        except Exception:  # Continue the batch when an individual pair cannot be processed.
            failed += 1  # Count the unsuccessful pair.
            LOG.exception("Failed pair: %s", name)  # Include the exception traceback for troubleshooting.

    LOG.info("Done: %d created, %d existing, %d incomplete, %d failed", created, skipped, incomplete, failed)  # Report all outcomes.

    return 1 if failed or incomplete else 0  # Signal partial failure if any discovered group was not handled successfully.


if __name__ == "__main__":  # Run the batch only when executed directly, not when imported.
    raise SystemExit(main())  # Return the batch result as the process exit code.
