#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create binary flood rasters from already processed NISAR dB GeoTIFFs.

This is the standalone flood-mask portion of ``NISAR_HV_Flood_Process.py``.
It does not open a NISAR HDF5/NetCDF product or repeat the SAR processing.

With no command-line arguments, every GeoTIFF below ``GeoTIFF_Processed`` is
used as input and the masks are written to ``Flood_Raster``. A pixel is flood
(``1``) when its valid input value is strictly below the selected threshold
(default: -15 dB); all other pixels,
including input NoData, are non-flood (``0``).
"""

# Keep type annotations as strings until they are needed, avoiding eager imports.
from __future__ import annotations

# Parse command-line options such as the input path and flood threshold.
import argparse
# Record progress, warnings, and errors in the terminal and a log file.
import logging
# Atomically replace the temporary output with the completed GeoTIFF.
import os
# Send terminal log messages to the standard output stream.
import sys
# Add a timestamp to the log-file name.
from datetime import datetime
# Work with platform-independent filesystem paths.
from pathlib import Path
# Describe the generator returned by the tiled-window helper.
from typing import Iterator

# Perform fast array masking and threshold comparisons.
import numpy as np
# Read and write GeoTIFF rasters.
import rasterio
# Select nearest-neighbour resampling for categorical flood-mask overviews.
from rasterio.enums import Resampling
# Define a rectangular raster region to read or write at one time.
from rasterio.windows import Window


# Locate the directory containing this script, regardless of the launch directory.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
# Use the processed-raster directory beside this script when no input is supplied.
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "GeoTIFF_Processed"
# Save flood masks beside the script by default.
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "Flood_Raster"
# Store timestamped run logs beside the script.
LOG_DIRECTORY = SCRIPT_DIRECTORY / "NISAR_logs"

# Accept both common GeoTIFF filename extensions.
SUPPORTED_EXTENSIONS = (".tif", ".tiff")
# Classify valid dB pixels below this value as flooded unless the user overrides it.
DEFAULT_FLOOD_THRESHOLD_DB = -15.0
# Process rasters in 512-by-512-pixel tiles to limit memory use.
TILE_SIZE = 512
# Candidate pyramid scales for faster display of large output rasters.
OVERVIEW_FACTORS = (2, 4, 8, 16, 32)


def parse_arguments() -> argparse.Namespace:
    """Return user-supplied or default input, output, threshold, and overwrite options."""
    # Create the command-line parser and its one-sentence help description.
    parser = argparse.ArgumentParser(
        description="Create binary flood masks from processed dB GeoTIFF files."
    )
    # Allow a GeoTIFF file or a recursively scanned directory as the positional input.
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "A processed dB GeoTIFF or a directory to scan recursively "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
    )
    # Let the user choose where generated flood-mask GeoTIFFs are written.
    parser.add_argument(
        "--output-directory",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory for flood masks (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    # Let the user override the dB cutoff used to label flood pixels.
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_FLOOD_THRESHOLD_DB,
        help=(
            "Valid pixels strictly below this dB value are flood "
            f"(default: {DEFAULT_FLOOD_THRESHOLD_DB:g})"
        ),
    )
    # Add a switch that permits replacement of an already valid output GeoTIFF.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a valid flood raster if it already exists.",
    )
    # Parse the process arguments and return them as an argparse Namespace.
    return parser.parse_args()


def setup_logger(log_directory: Path) -> logging.Logger:
    """Configure and return a logger that writes to both a dated file and the terminal."""
    # Create the log directory and any missing parent directories.
    log_directory.mkdir(parents=True, exist_ok=True)
    # Make a unique, time-stamped filename for this run's log.
    log_path = log_directory / (
        f"NISAR_HV_Flood_From_GeoTIFF_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    # Retrieve the named logger so all messages from this script share configuration.
    logger = logging.getLogger("NISAR_HV_Flood_From_GeoTIFF")
    # Capture DEBUG and more severe messages before handlers apply their own level.
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate messages from being passed to the root logger.
    logger.propagate = False
    # Remove old handlers in case this module runs more than once in one process.
    logger.handlers.clear()

    # Use the same timestamped, level-labelled format for every destination.
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Configure one handler for the terminal and another for the log file.
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ):
        # Allow each handler to emit detailed diagnostic messages.
        handler.setLevel(logging.DEBUG)
        # Apply the common message format to the current handler.
        handler.setFormatter(formatter)
        # Attach the configured handler to this script's logger.
        logger.addHandler(handler)

    # Record the exact location of the log file for the user.
    logger.info("Log file: %s", log_path)
    # Give the fully configured logger to the caller.
    return logger


def source_windows(height: int, width: int) -> Iterator[Window]:
    """Yield windows that cover a raster in fixed-size tiles without loading it all at once."""
    # Step down the raster rows in TILE_SIZE-pixel increments.
    for row_offset in range(0, height, TILE_SIZE):
        # Shorten the final row tile if the height is not an exact multiple of TILE_SIZE.
        tile_height = min(TILE_SIZE, height - row_offset)
        # Step across the raster columns in TILE_SIZE-pixel increments.
        for column_offset in range(0, width, TILE_SIZE):
            # Shorten the final column tile if the width is not an exact multiple of TILE_SIZE.
            tile_width = min(TILE_SIZE, width - column_offset)
            # Yield the current tile as rasterio's column, row, width, height window.
            yield Window(column_offset, row_offset, tile_width, tile_height)


def discover_geotiffs(input_path: Path) -> list[Path]:
    """Return a single valid GeoTIFF or every eligible GeoTIFF below a directory."""
    # Handle an explicitly supplied file without scanning a directory.
    if input_path.is_file():
        # Reject files whose extension is not one of the supported GeoTIFF extensions.
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Input file is not a .tif or .tiff file: {input_path}")
        # Return the requested file in a list so callers use one input interface.
        return [input_path]

    # Stop with a clear error if the path is neither a file nor a directory.
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Recursively find eligible GeoTIFFs and sort them for deterministic processing.
    return sorted(
        path
        # Visit every descendant of the selected input directory.
        for path in input_path.rglob("*")
        # Keep ordinary files only.
        if path.is_file()
        # Keep files with a supported extension, case-insensitively.
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        # Ignore incomplete temporary outputs from interrupted previous runs.
        and not path.name.lower().endswith(".part.tif")
        # Avoid treating previously created flood masks as new source rasters.
        and "_flood_raster_" not in path.name.lower()
    )


def threshold_label(threshold_db: float) -> str:
    """Return a compact, filesystem-friendly string representation of the threshold."""
    # Use general numeric formatting to remove unnecessary trailing zeroes.
    return f"{threshold_db:g}"


def flood_raster_path(
    processed_path: Path,
    output_directory: Path,
    threshold_db: float,
) -> Path:
    """Build the output flood-mask filename from the source name and threshold."""
    # Preserve the source basename, add the flood rule, and place it in the output directory.
    return output_directory / (
        f"{processed_path.stem}_Flood_Raster_lt_{threshold_label(threshold_db)}dB.tif"
    )


def valid_existing_output(output_path: Path) -> bool:
    """Return True only for a readable, nonempty, one-band uint8 GeoTIFF."""
    # A missing path cannot be reused as an existing output.
    if not output_path.is_file():
        return False
    try:
        # Open the candidate output only long enough to inspect its metadata.
        with rasterio.open(output_path) as raster:
            # Verify the expected format, shape, band count, and categorical data type.
            return (
                raster.driver == "GTiff"
                and raster.count == 1
                and raster.width > 0
                and raster.height > 0
                and raster.dtypes[0] == "uint8"
            )
    # Treat unreadable or malformed rasters as invalid rather than aborting the run.
    except (OSError, rasterio.errors.RasterioError):
        return False


def export_flood_raster(
    processed_path: Path,
    output_directory: Path,
    threshold_db: float,
    overwrite: bool,
    logger: logging.Logger,
) -> tuple[Path, bool]:
    """Create one binary flood mask and return its path plus whether it was created."""
    # Derive the final filename before deciding whether work is needed.
    output_path = flood_raster_path(processed_path, output_directory, threshold_db)
    # Reuse a complete existing file unless the caller explicitly requests replacement.
    if not overwrite and valid_existing_output(output_path):
        logger.info("Skipping; flood raster already exists: %s", output_path)
        return output_path, False

    # Ensure the selected output directory exists before creating a raster inside it.
    output_directory.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so incomplete results are never mistaken for outputs.
    temp_path = output_path.with_suffix(".part.tif")
    # Remove a stale temporary file left behind by an interrupted earlier attempt.
    if temp_path.exists():
        temp_path.unlink()

    try:
        # Open the processed dB GeoTIFF as the source for thresholding.
        with rasterio.open(processed_path) as source_raster:
            # Require a nonempty first band before creating an output.
            if source_raster.count < 1 or source_raster.width < 1 or source_raster.height < 1:
                raise ValueError("input does not contain a nonempty raster band")

            # Start with the source spatial metadata so output pixels remain aligned.
            profile = source_raster.profile.copy()
            # Replace source-specific properties with binary-flood-mask output properties.
            profile.update(
                {
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1,
                    "nodata": None,
                    "compress": "deflate",
                    "predictor": 2,
                    "BIGTIFF": "IF_SAFER",
                }
            )

            # Create the one-band uint8 temporary GeoTIFF using the configured profile.
            with rasterio.open(temp_path, "w", **profile) as flood_raster:
                # Describe what band 1 contains for GIS users.
                flood_raster.set_band_description(
                    1, f"Flood mask: dB < {threshold_db:g}"
                )
                # Embed source and classification details as GeoTIFF metadata tags.
                flood_raster.update_tags(
                    source_geotiff=str(processed_path.resolve()),
                    flood_threshold_db=str(threshold_db),
                    threshold_operator="<",
                    flood_pixel_value="1",
                    non_flood_pixel_value="0",
                )

                # Calculate the number of read/write tiles to report percentage progress.
                total_tiles = (
                    (source_raster.height + TILE_SIZE - 1) // TILE_SIZE
                ) * ((source_raster.width + TILE_SIZE - 1) // TILE_SIZE)

                # Process each source tile independently to constrain memory consumption.
                for tile_number, window in enumerate(
                    source_windows(source_raster.height, source_raster.width), start=1
                ):
                    # Read band 1 and preserve its NoData mask for the current window.
                    db_tile = source_raster.read(1, window=window, masked=True)
                    # Mark only valid, finite dB values strictly below the threshold as flood.
                    flood_pixels = (
                        ~np.ma.getmaskarray(db_tile)
                        & np.isfinite(db_tile.data)
                        & (db_tile.data < threshold_db)
                    )
                    # Convert Boolean flood labels to 0/1 uint8 values and write this tile.
                    flood_raster.write(
                        flood_pixels.astype(np.uint8), 1, window=window
                    )

                    # Log progress at roughly ten-percent intervals and on the final tile.
                    if (
                        tile_number == total_tiles
                        or tile_number % max(1, total_tiles // 10) == 0
                    ):
                        # Report the input name, percentage, and tile counts.
                        logger.info(
                            "%s progress: %.0f%% (%d/%d tiles)",
                            processed_path.name,
                            100 * tile_number / total_tiles,
                            tile_number,
                            total_tiles,
                        )

                # Keep only overview scales that fit within both raster dimensions.
                valid_overview_factors = [
                    factor
                    for factor in OVERVIEW_FACTORS
                    if factor <= min(flood_raster.width, flood_raster.height)
                ]
                # Build display overviews only when at least one valid scale exists.
                if valid_overview_factors:
                    # Use nearest-neighbour resampling so the 0/1 class values remain categorical.
                    flood_raster.build_overviews(
                        valid_overview_factors, Resampling.nearest
                    )
                    # Record the resampling method in Rasterio's overview metadata namespace.
                    flood_raster.update_tags(ns="rio_overview", resampling="nearest")

        # Atomically promote the completed temporary file to its final filename.
        os.replace(temp_path, output_path)
    # Clean up a partial temporary file if opening, reading, or writing failed.
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    # Record successful output creation with the applied threshold.
    logger.info(
        "Created binary flood raster (dB < %g): %s", threshold_db, output_path
    )
    return output_path, True


def main() -> None:
    """Run the command-line workflow for one input file or an input directory."""
    # Read command-line options once at startup.
    arguments = parse_arguments()
    # Start terminal and file logging before validating or processing inputs.
    logger = setup_logger(LOG_DIRECTORY)

    try:
        # Resolve the input path and discover the GeoTIFF files to process.
        source_files = discover_geotiffs(arguments.input_path.resolve())
    # Turn user-correctable path and extension errors into a nonzero command exit.
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    # Finish successfully when the input directory contains no eligible GeoTIFFs.
    if not source_files:
        logger.warning("No GeoTIFF inputs found below: %s", arguments.input_path)
        return

    # Resolve the output location once before it is passed to every export operation.
    output_directory = arguments.output_directory.resolve()
    # Summarize how many inputs will be processed and the flood classification rule.
    logger.info(
        "Found %d processed GeoTIFF(s); threshold is dB < %g.",
        len(source_files),
        arguments.threshold_db,
    )

    # Count newly created outputs for the final run summary.
    created_count = 0
    # Count outputs reused because they were already valid.
    skipped_count = 0
    # Count inputs that could not be converted without stopping the other inputs.
    failed_count = 0
    # Process every discovered input independently so one bad raster does not halt the batch.
    for source_file in source_files:
        try:
            # Write or reuse this input's companion flood raster.
            _, created = export_flood_raster(
                source_file,
                output_directory,
                arguments.threshold_db,
                arguments.overwrite,
                logger,
            )
            # Update the appropriate result count according to the export status.
            if created:
                created_count += 1
            else:
                skipped_count += 1
        # Log an expected per-file processing failure and continue with the next input.
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            failed_count += 1
            logger.error("Failed to process %s: %s", source_file, exc)
            logger.debug("Detailed traceback:", exc_info=True)

    # Write a complete created/skipped/failed summary after all inputs are attempted.
    logger.info(
        "Flood export complete: %d created, %d skipped, %d failed.",
        created_count,
        skipped_count,
        failed_count,
    )
    # Return a nonzero status when any source file failed, enabling automation to detect it.
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    # Run the command-line workflow only when this file is executed directly.
    main()
