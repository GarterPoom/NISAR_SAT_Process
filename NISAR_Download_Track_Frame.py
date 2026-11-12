"""
NISAR_Download_Track_Frame.py

Search NASA's ASF (Alaska Satellite Facility) catalog for NISAR granules
within a given area of interest, date range and Track/Frame selection, and download
*every* file of each granule – the main HDF5 product plus all metadata (QA_STATS.h5,
ISO XML, JSON/YAML, QA reports) and visualizations (PNG browse images, KML footprints).
All files of one granule are saved together in ``<OUTPUT_DIRECTORY>/<granule_id>/``.

Each file download shows its own byte-level progress bar (current bytes / total bytes),
rather than a single progress bar tracking file count.

Requirements:
    pip install asf_search tqdm requests geopandas shapely python-dotenv

Credentials:
    Copy .env.example to .env and set EARTHDATA_USERNAME and
    EARTHDATA_PASSWORD there before running the script. ".env" is git-ignored.

Usage:
    python NISAR_Download_Track_Frame.py
"""  # End of module‑level docstring – describes the whole script.
# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).
import logging  # Standard logging module for recording execution steps, warnings, and errors.
import re  # Regular-expression support for confirming Track/Frame values in returned filenames.
import time  # Monotonic speed measurements and retry backoff delays.
from datetime import datetime, timedelta  # Date/time tools for configured and rolling search windows.
from urllib.parse import unquote, urlparse  # Safely extracts filenames from download URLs.

import requests  # HTTP library; used here for its exceptions and streamed GET requests.
from tqdm import tqdm  # Library for rendering dynamic progress bars in the terminal console.
import asf_search as asf  # Alaska Satellite Facility Search Python package, imported under alias 'asf'.
import geopandas as gpd  # Library for reading shapefiles and handling geospatial vector data.
from dotenv import load_dotenv  # Loads Earthdata credentials from the git-ignored .env file.

# Read .env from the script's own folder so credentials load regardless of the working directory.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# --------------------------------------------------------------------------- #
# Configuration – all tunable settings are gathered in this class.
# --------------------------------------------------------------------------- #
class Config:  # Groups every tunable setting in one place instead of scattering local variables.
    """Central configuration for the search-and-download workflow.

    Keeping these values in one place makes the script easier to adapt
    (e.g., for a different AOI, date range, or product level) without
    hunting through function bodies.
    """
    # Loaded from the git-ignored .env file (see .env.example); never store credentials in source control.
    EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")  # NASA Earthdata login username.
    EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")  # NASA Earthdata login password.

    LOG_DIRECTORY = "NISAR_Download_logs"  # Folder where timestamped log files are written.
    OUTPUT_DIRECTORY = "NISAR_Product"  # Each granule's files are saved together in <OUTPUT_DIRECTORY>/<granule_id>/.
    INCLUDE_PRIVATE_FILES = False  # Files in the NISAR-JPL-PRIVATE-DATA bucket need special access.

    # Area of interest, as a path to a shapefile (.shp). All features in the
    # file are dissolved into a single geometry and reprojected to WGS84
    # (EPSG:4326) automatically before being sent to the ASF search API.

    # Directory where this .py file lives
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Build the full path relative to that directory
    AOI_SHAPEFILE = os.path.join(script_dir,
                             "Thailand_Admin",  # Subfolder holding the Thailand admin boundary shapefile.
                             "tha_admbnda_adm1_rtsd_20190221.shp")  # Path to the shapefile defining the area of interest (AOI).

    # Mode 1 uses the manually configured dates below. Mode 2 ignores them and
    # searches from DATE_LOOKBACK_DAYS ago through the current date and time.
    DATE_MODE = 1  # Select date mode: 1 = manual START/END_DATE, 2 = rolling lookback window.
    START_DATE = datetime.strptime("2026-11-05", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    END_DATE = datetime.strptime("2026-11-06", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    DATE_LOOKBACK_DAYS = 10  # Number of days to look back when DATE_MODE is 2.

    PRODUCT_LEVEL = "GSLC"  # NISAR processing level to filter results by.
    FRAME_COVERAGE = "FULL"  # Exclude NISAR products that cover only a partial frame.
    POLARIZATION_MODE = "DH"  # "SH" (single-pol H), "DH" (dual-pol H: HH+HV), or None to keep every mode.

    # NISAR Track/Frame pairs to download.  Add every required pair here as
    # (track, frame); for example, (105, 78) means Track 105, Frame 078.
    # A pair is downloaded only when its full-frame product footprint
    # intersects AOI_SHAPEFILE; partial-frame products are excluded.
    # Leave no pairs configured only if you want the script to stop before
    # searching, rather than accidentally downloading every AOI result.
    TRACK_FRAME_PAIRS: list[tuple[int, int]] = [  # List of (track, frame) integer pairs to search for.
        (4, 81), (4, 82), (11, 8), (11, 9), (11, 10)  # Configured Track/Frame pairs for the Thailand AOI.
    ]  # End of TRACK_FRAME_PAIRS list.

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 256 * 1024  # 256 KB chunks keep progress and slow-link checks responsive.
    DOWNLOAD_CONNECT_TIMEOUT = 30  # Seconds allowed to establish an HTTP connection.
    DOWNLOAD_READ_TIMEOUT = 180  # Seconds allowed without receiving any download data.
    DOWNLOAD_MIN_SPEED = 16 * 1024  # Retry when sustained throughput falls below 16 KiB/s; set to 0 to disable.
    DOWNLOAD_SPEED_WINDOW = 300  # Seconds over which sustained low throughput is measured.
    DOWNLOAD_MAX_ATTEMPTS = 8  # Initial attempt plus retries for interrupted or persistently slow downloads.
    DOWNLOAD_RETRY_BACKOFF = 5  # Base seconds between retries; doubles after each failure.
    DOWNLOAD_MAX_BACKOFF = 120  # Cap retry delays so recovery does not pause for an excessive period.


def resolve_date_range(  # Resolve the search date range from the configured date mode.
    date_mode: int,        # Input parameter: integer mode selector (1 for manual, 2 for rolling)
    start_date: datetime,  # Input parameter: user-specified start datetime object
    end_date: datetime,    # Input parameter: user-specified end datetime object
    lookback_days: int,    # Input parameter: number of days to look back from today
) -> tuple[datetime, datetime]:  # Return type hint: returns a tuple containing two datetime objects
    """
    Calculates and returns a (start_date, end_date) pair based on the selected mode.

    Modes:
        - Mode 1 (Manual): Returns user-provided start_date and end_date.
        - Mode 2 (Rolling): Sets end_date to current time and start_date to
          (current time - lookback_days).

    Parameters:
        date_mode (int): 1 for manual dates, 2 for a rolling date window.
        start_date (datetime): Manual start date (required if date_mode == 1).
        end_date (datetime): Manual end date (required if date_mode == 1).
        lookback_days (int): Days to look back from current time (required if date_mode == 2).

    Returns:
        tuple[datetime, datetime]: A tuple containing (resolved_start, resolved_end).

    Raises:
        ValueError: If date_mode is invalid, lookback_days is invalid, start/end dates
                    are not datetime instances, or start_date occurs after end_date.
    """
    # Check if date_mode is invalid (must be 1 or 2, and must NOT be a boolean)
    if isinstance(date_mode, bool) or date_mode not in (1, 2):
        # Raise an exception stopping execution if the mode is invalid
        raise ValueError("DATE_MODE must be 1 (manual dates) or 2 (rolling dates).")

    # Evaluate execution path based on the validated date_mode
    if date_mode == 1:
        # Assign manual start_date and end_date to local variables
        resolved_start, resolved_end = start_date, end_date
        # Define description string used later in log output
        mode_description = "manual"
    else:
        # Validate lookback_days for rolling mode (must be non-boolean, integer, and >= 0)
        if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 0:
            # Raise an exception if lookback_days does not meet requirements
            raise ValueError("DATE_LOOKBACK_DAYS must be a non-negative integer.")

        # Get the current system date and time for the rolling end date
        resolved_end = datetime.now()
        # Compute start date by subtracting lookback_days from resolved_end
        resolved_start = resolved_end - timedelta(days=lookback_days)
        # Define dynamic description string incorporating lookback_days count
        mode_description = f"rolling {lookback_days}-day"

    # Ensure resolved dates are valid datetime instances
    if not isinstance(resolved_start, datetime) or not isinstance(resolved_end, datetime):
        # Raise an exception if input manual dates were not datetime objects
        raise ValueError("START_DATE and END_DATE must be datetime values.")

    # Check if the start date occurs after the end date
    if resolved_start > resolved_end:
        # Raise an exception to prevent logically invalid date ranges
        raise ValueError("START_DATE must not be later than END_DATE.")

    # Write informative log entry using the formatted date strings (%Y-%m-%d %H:%M:%S)
    logging.info(
        "Using %s date mode: %s to %s", # Log template string
        mode_description,               # Replaces 1st %s (e.g., "manual" or "rolling 7-day")
        resolved_start.strftime("%Y-%m-%d %H:%M:%S"),  # Replaces 2nd %s with formatted start date
        resolved_end.strftime("%Y-%m-%d %H:%M:%S"),    # Replaces 3rd %s with formatted end date
    )  # End of logging call.

    # Return the final computed start and end datetimes as a tuple
    return resolved_start, resolved_end
# --------------------------------------------------------------------------- #
# Logging setup – configures logging to write to both a timestamped log file and stdout.
# --------------------------------------------------------------------------- #
def setup_logging(log_directory: str) -> str:  # Configure logging to write to both a timestamped log file and stdout.
    """Configure logging to write to both a timestamped log file and stdout.

    Args:
        log_directory: Directory where the log file will be created
            (created automatically if it doesn't already exist).

    Returns:
        The full path to the created log file.
    """
    os.makedirs(log_directory, exist_ok=True)  # Create log folder safely without raising errors if it exists.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format current date/time as a timestamp string.
    log_filename = f"nisar_search_download_{timestamp}.log"  # Build a dynamic, timestamped log filename.
    log_filepath = os.path.join(log_directory, log_filename)  # Construct the full path for the log file.

    logging.basicConfig(  # Initialize the global logging configuration.
        level=logging.INFO,  # Set minimum logging threshold to INFO level.
        format="%(asctime)s [%(levelname)s] %(message)s",  # Set timestamped format for log output lines.
        handlers=[  # Specify list of active log message destinations.
            logging.FileHandler(log_filepath),  # Handler 1: save log output to the log file.
            logging.StreamHandler(sys.stdout),  # Handler 2: display log output on standard stdout,
        ],  # End of handlers list.
    )  # End of logging.basicConfig() call.

    logging.info(f"Initialized logging session. File path: {os.path.abspath(log_filepath)}")  # Write initial log entry.
    return log_filepath  # Return the full path of the created log file.

# --------------------------------------------------------------------------- #
# Authentication – logs in to NASA Earthdata and returns an active ASFSession.
# --------------------------------------------------------------------------- #
def authenticate_earthdata(username: str, password: str) -> asf.ASFSession:  # Authenticate with NASA Earthdata and return an active session.
    """Authenticate with NASA Earthdata and return an active session.

    Args:
        username: NASA Earthdata login username.
        password: NASA Earthdata login password.

    Returns:
        An authenticated ASFSession, usable for both searching and
        downloading (it carries the auth cookies needed for direct
        HTTP requests too).

    Exits:
        If authentication fails.
    """
    if not username or not password:  # Check that both Earthdata credentials were loaded from .env.
        logging.error(  # Log an error explaining how to supply the missing credentials.
            "NASA Earthdata credentials are missing. Set EARTHDATA_USERNAME "  # First part of the error message.
            "and EARTHDATA_PASSWORD in the .env file (see .env.example)."  # Second part of the error message.
        )  # End of error message call.
        sys.exit(1)  # Stop the script with a failure status code.

    session = asf.ASFSession()  # Create an unauthenticated ASFSession instance.
    try:  # Begin try block for the Earthdata authentication attempt.
        session.auth_with_creds(username, password)  # Authenticate the session using the provided credentials.
        logging.info("Successfully authenticated with NASA Earthdata.")  # Log successful authentication.
        return session  # Return the authenticated session object.
    except Exception as auth_error:  # Intercept any authentication exceptions.
        logging.error(f"Authentication failed: {auth_error}")  # Log authentication failure details.
        sys.exit(1)  # Exit script execution with a failure status code.

# --------------------------------------------------------------------------- #
# AOI loading – reads a shapefile and returns a single WKT string for the AOI.
# --------------------------------------------------------------------------- #
def load_aoi_wkt_from_shapefile(shapefile_path: str) -> str:  # Read a shapefile and convert its geometry to a single WKT string.
    """Read a shapefile and convert its geometry to a single WKT string.

    Handles the details asf_search's `intersectsWith` option needs but a
    raw shapefile doesn't guarantee on its own:
      - Reprojects to WGS84 (EPSG:4326) if the shapefile uses a different
        coordinate reference system, since ASF expects lon/lat degrees.
      - Dissolves multiple features/polygons into one combined geometry,
        so a multi-polygon shapefile still produces a single valid AOI.

    Args:
        shapefile_path: Path to the .shp file (its sibling .shx/.dbf/.prj
            files must sit alongside it, as is standard for shapefiles).

    Returns:
        A WKT geometry string representing the (possibly combined) AOI.

    Exits:
        If the shapefile can't be read, contains no features, or its
        geometry can't be converted to WKT.
    """
    logging.info(f"Loading AOI from shapefile: {os.path.abspath(shapefile_path)}")  # Log the shapefile path being read.

    try:  # Begin try block for reading and processing the shapefile.
        aoi_gdf = gpd.read_file(shapefile_path)  # Read the shapefile into a GeoDataFrame.

        if aoi_gdf.empty:  # Check whether the shapefile contained any features at all.
            logging.error("Shapefile contains no features. Exiting script.")  # Log the empty-file error.
            sys.exit(1)  # Exit script execution with a failure status code.

        if aoi_gdf.crs is None:  # Check whether the shapefile has a defined coordinate reference system.
            logging.warning(  # Warn that a missing CRS is being assumed to already be WGS84.
                "Shapefile has no defined CRS; assuming it is already WGS84 (EPSG:4326)."  # Warning message text.
            )  # End of logging call.
            aoi_gdf = aoi_gdf.set_crs(epsg=4326)  # Explicitly tag the GeoDataFrame as WGS84 without reprojecting values.
        elif aoi_gdf.crs.to_epsg() != 4326:  # Check whether the CRS is something other than WGS84.
            logging.info(f"Reprojecting AOI from {aoi_gdf.crs} to EPSG:4326 (WGS84).")  # Log the reprojection being applied.
            aoi_gdf = aoi_gdf.to_crs(epsg=4326)  # Reproject all geometries to WGS84 lon/lat.

        combined_geometry = aoi_gdf.union_all()  # Dissolve every feature's geometry into a single combined geometry.

        if combined_geometry.is_empty:  # Check whether the dissolved geometry ended up empty.
            logging.error("Combined AOI geometry is empty after processing. Exiting script.")  # Log the empty-geometry error.
            sys.exit(1)  # Exit script execution with a failure status code.

        aoi_wkt = combined_geometry.wkt  # Convert the combined shapely geometry to a WKT string.
        logging.info(f"Successfully derived AOI WKT from shapefile ({len(aoi_gdf)} feature(s) combined).")  # Log success and feature count.
        return aoi_wkt  # Return the WKT string for use as the search's spatial filter.

    except Exception as shapefile_error:  # Intercept any error reading or processing the shapefile.
        logging.error(f"Failed to load AOI from shapefile '{shapefile_path}': {shapefile_error}")  # Log the detailed error.
        sys.exit(1)  # Exit script execution with a failure status code.

# --------------------------------------------------------------------------- #
# Search – queries the ASF catalog for NISAR granules matching the given filters.
# --------------------------------------------------------------------------- #
def prepare_track_frame_filters(  # Validate Track/Frame pairs and build filename search patterns.
    track_frame_pairs: list[tuple[int, int]],  # Input: list of (track, frame) tuples from Config.
) -> tuple[set[tuple[int, int]], list[str]]:  # Returns: (set of validated pairs, list of granule name patterns).
    """Validate Track/Frame pairs and create NISAR filename search patterns.

    ASF documents that NISAR currently lacks searchable Track/Frame metadata.
    NISAR products do, however, encode them in their granule names as
    ``..._<track>_<direction>_<frame>_...``.  ``granule_list`` patterns are
    therefore used for the Track/Frame part of the search, while
    ``intersectsWith`` remains the spatial AOI filter.
    """
    if not track_frame_pairs:  # Stop early when no Track/Frame pairs are configured.
        logging.error(  # Log an error explaining how to configure pairs.
            "No TRACK_FRAME_PAIRS configured. Add one or more (track, frame) "  # First part of the error message.
            "pairs in Config before running the script."  # Second part of the error message.
        )  # End of error message call.
        sys.exit(1)  # Exit with failure so nothing is downloaded unfiltered.

    selected_pairs: set[tuple[int, int]] = set()  # Collect unique, validated (track, frame) pairs.
    for pair in track_frame_pairs:  # Iterate over every configured pair.
        if not isinstance(pair, tuple) or len(pair) != 2:  # Each entry must be a tuple of exactly two items.
            logging.error(  # Log the malformed-entry error.
                "Each TRACK_FRAME_PAIRS entry must be a two-item tuple "  # First part of the error message.
                "such as (105, 78)."  # Second part of the error message.
            )  # End of error message call.
            sys.exit(1)  # Exit with failure on a malformed entry.

        track, frame = pair  # Unpack the pair into track and frame numbers.
        if (  # Start of validation condition for track and frame values.
            isinstance(track, bool)  # Reject booleans (bool is a subclass of int).
            or isinstance(frame, bool)  # Reject booleans for frame as well.
            or not isinstance(track, int)  # Track must be an integer.
            or not isinstance(frame, int)  # Frame must be an integer.
            or track < 0  # Track must not be negative.
            or frame < 0  # Frame must not be negative.
        ):  # End of validation condition.
            logging.error(  # Log the invalid-pair error.
                f"Invalid Track/Frame pair {pair!r}. Both values must be non-negative integers."  # Error message naming the invalid pair.
            )  # End of error message call.
            sys.exit(1)  # Exit with failure on an invalid pair.
        selected_pairs.add((track, frame))  # Store the validated pair (duplicates collapse automatically).

    # The `?` is the ascending/descending direction character between Track
    # and Frame.  The leading `*` accommodates the NISAR product/version
    # fields that precede the Track in the granule name.
    granule_patterns = [  # Build one granule filename pattern per pair.
        f"NISAR_*{track:03d}_?_{frame:03d}_*"  # Pattern with zero-padded 3-digit track and frame numbers.
        for track, frame in sorted(selected_pairs)  # Iterate over pairs in sorted order for deterministic output.
    ]  # End of list comprehension.
    return selected_pairs, granule_patterns  # Return validated pairs and their search patterns.


def search_nisar_granules(  # Query the ASF catalog for NISAR granules matching the given filters.
    aoi_wkt: str,  # Area of interest as a WKT geometry string.
    start_date: datetime,  # Earliest acquisition date to include.
    end_date: datetime,  # Latest acquisition date to include.
    product_level: str,  # NISAR processing level to filter on (e.g. "GSLC").
    frame_coverage: str,  # NISAR frame coverage to include ("FULL" excludes partial scenes).
    granule_patterns: list[str],  # NISAR filename patterns for selected Track/Frame pairs.
    max_results: int,  # Maximum number of granules to return.

) -> asf.ASFSearchResults:  # The raw ASF search results object.
    """Query the ASF catalog for NISAR granules matching the given filters.

    Args:
        aoi_wkt: Area of interest as a WKT geometry string.
        start_date: Earliest acquisition date to include.
        end_date: Latest acquisition date to include.
        product_level: NISAR processing level to filter on (e.g. "GSLC").
        frame_coverage: NISAR frame coverage to include; use ``"FULL"`` to
            exclude partial-frame products.
        granule_patterns: NISAR filename patterns representing the selected
            Track/Frame pairs.
        max_results: Maximum number of granules to return.

    Returns:
        The raw ASF search results object.

    Exits:
        If the search request fails.
    """
    logging.info(f"Area of Interest (AOI WKT): {aoi_wkt}")  # Log the specified spatial coverage WKT boundary.
    logging.info(f"Search date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")  # Log the search time range.
    logging.info(f"Target Processing Level: {product_level}")  # Log the selected target product level.
    logging.info(f"Target Frame Coverage: {frame_coverage}")  # FULL prevents partial-frame products from being returned.
    logging.info(  # Log the Track/Frame patterns used in the search.
        "Selected Track/Frame granule pattern(s): " + ", ".join(granule_patterns)  # Join patterns into one comma-separated string.
    )  # End of logging call.

    search_options = asf.ASFSearchOptions(  # Initialize the ASF search configuration object.
        dataset=["NISAR"],  # Filter search results to the NISAR dataset platform.
        # Spatially select frames that overlap the AOI. frameCoverage below
        # independently rejects NISAR products containing only part of a frame.
        intersectsWith=aoi_wkt,  # Spatial filter: AOI WKT geometry.
        # NISAR Track/Frame is encoded in its filename. This works with
        # NISAR even where CMR has no searchable Track/Frame metadata.
        granule_list=granule_patterns,  # Track/Frame filter: filename wildcard patterns.
        start=start_date,  # Filter granules acquired on or after the start date.
        end=end_date,  # Filter granules acquired on or before the end date.
        processingLevel=[product_level],  # Filter granules by the specified product level.
        frameCoverage=frame_coverage,  # Retain full-frame products and exclude partial scenes.
        maxResults=max_results,  # Limit the maximum number of search results retrieved.
    )  # End of ASFSearchOptions().

    logging.info("Querying ASF search API for matching NISAR datasets...")  # Log the start of the query.
    try:  # Begin try block for the API search request.
        results = asf.search(opts=search_options)  # Execute the catalog search query against the ASF API.
        logging.info(f"Search completed. Found {len(results)} matching granules.")  # Log the total hit count.
        return results  # Return the search results object.
    except Exception as search_error:  # Intercept any search query errors.
        logging.error(f"Search query failed: {search_error}")  # Log search query failure details.
        sys.exit(1)  # Exit script execution with a failure status code.

# --------------------------------------------------------------------------- #
# Filter – groups every file (HDF5, metadata, visualizations) per granule.
# --------------------------------------------------------------------------- #
def normalize_polarization_mode(polarization_mode: str | None) -> str | None:  # Normalize/validate the configured polarization mode.
    """Return an upper-case polarization mode code, or ``None`` for no filtering.

    Args:
        polarization_mode: ``"SH"`` (single-pol H) or ``"DH"`` (dual-pol H); ``None``/``"ANY"`` keeps every mode.

    Raises:
        ValueError: If the value is not SH, DH, ANY, or None.
    """
    if polarization_mode is None or str(polarization_mode).strip().upper() in ("", "ANY"):  # None, empty or ANY means keep every mode.
        return None  # No polarization filtering requested.
    mode = str(polarization_mode).strip().upper()  # Normalize case and whitespace.
    if mode not in ("SH", "DH"):  # Only SH and DH are supported.
        raise ValueError("POLARIZATION_MODE must be 'SH', 'DH', or None.")  # Reject unsupported modes early.
    return mode  # Return the validated mode code.


def polarization_mode_from_filename(filename: str) -> str | None:  # Read the polarization mode code from a NISAR filename.
    """Extract the frequency-A polarization mode (e.g. ``DH``) from a NISAR filename.

    NISAR names embed ``_<bandwidth>_<modeA><modeB>_`` such as ``_2005_DHDH_``;
    the first two letters describe the frequency-A polarization mode.
    """
    match = re.search(r"_\d{4}_([A-Z]{2})([A-Z]{2})_", filename)  # Locate the 4-letter mode token after the bandwidth code.
    return match.group(1) if match else None  # None when the name has no recognizable mode token.


# Capture NISAR's ..._<track>_<direction>_<frame>_... granule-name fields.
TRACK_FRAME_PATTERN = re.compile(r"^NISAR_L2_[^_]+_[^_]+_\d+_(\d+)_[AD]_(\d+)_")  # Regex groups: (1) track number, (2) frame number.

# ASF lists some files from a JPL-internal bucket that ordinary Earthdata accounts cannot access.
PRIVATE_JPL_BUCKET = "/NISAR-JPL-PRIVATE-DATA/"  # URL fragment identifying JPL-private files.


def get_product_urls(product: asf.ASFProduct) -> list[str]:  # Collect all download URLs for one ASF product.
    """Return every HTTPS URL ASF lists for one product.

    Covers the main HDF5 file plus ``additionalUrls`` (QA_STATS, ISO XML,
    KML, JSON/YAML metadata, QA reports) and browse images (PNG).
    """
    if hasattr(product, "find_urls"):  # Newer asf_search versions provide find_urls().
        urls = list(product.find_urls())  # Use the library helper to gather every URL.
    else:  # Older asf_search versions without ASFProduct.find_urls().
        properties = product.properties  # Fall back to the raw product properties.
        urls = [properties.get("url")]  # Start with the main product URL.
        urls.extend(properties.get("additionalUrls") or [])  # Add metadata/QA URLs, if any.
        browse = properties.get("browse") or []  # Browse images may be a string or a list.
        urls.extend([browse] if isinstance(browse, str) else browse)  # Normalize browse to a list and add it.
    return [url for url in urls if url]  # Drop empty/None URLs.


def group_granule_urls(  # Group search results into per-granule URL lists.
    results: asf.ASFSearchResults,  # Search results returned by search_nisar_granules().
    selected_track_frame_pairs: set[tuple[int, int]],  # Exact (track, frame) pairs to keep.
    polarization_mode: str | None = None,  # Optional polarization mode filter (SH/DH).
    include_private_files: bool = False,  # Whether to keep JPL-private bucket files.
) -> dict[str, list[str]]:  # Returns {granule_id: [file URLs]}.
    """Group search results into ``{granule_id: [every file URL of that granule]}``.

    Every file belonging to a granule is kept – the main HDF5 product, all
    metadata (QA_STATS.h5, ISO XML, JSON, YAML, logs, QA reports) and all
    visualizations (PNG browse images, KML footprints).  Non-file links such
    as the ``s3credentials`` endpoint are dropped because their filenames do
    not start with the granule ID.

    Args:
        results: Search results returned by search_nisar_granules().
        selected_track_frame_pairs: Exact (track, frame) pairs to keep.
        polarization_mode: Optional ``"SH"`` or ``"DH"``; granules with another mode are dropped.
        include_private_files: Whether to keep files from the NISAR-JPL-PRIVATE-DATA bucket.

    Returns:
        Ordered mapping of granule ID (no extension) to its de-duplicated URLs.

    Exits:
        Gracefully (status 0) if no matching granules are found.
    """
    polarization_mode = normalize_polarization_mode(polarization_mode)  # Validate the requested mode.
    granule_urls: dict[str, list[str]] = {}  # Accumulator: granule ID -> list of file URLs.
    total_urls = 0  # Counter of all URLs seen across all products.

    for product in results:  # Iterate over every product in the search results.
        granule_id = product.properties.get("sceneName") or product.properties.get("fileID")  # Use sceneName, falling back to fileID, as the granule ID.
        if not granule_id:  # Skip results with no usable identifier.
            logging.warning("Skipping a search result without a sceneName/fileID.")  # Warn about the unidentifiable result.
            continue  # Move on to the next product.
        if granule_id.lower().endswith((".h5", ".hdf5")):  # Strip an HDF5 extension if the ID includes one.
            granule_id = os.path.splitext(granule_id)[0]  # Keep only the base name without extension.

        # This local check is deliberately kept in addition to the server-side
        # `granule_list` filter, so only exact requested pairs are downloaded.
        track_frame_match = TRACK_FRAME_PATTERN.match(granule_id)  # Parse track and frame from the granule name.
        if track_frame_match is None:  # Name did not match the expected NISAR pattern.
            logging.warning(f"Skipping granule with an unrecognised NISAR name: {granule_id}")  # Warn about the unrecognised name.
            continue  # Skip this product.
        track_frame_pair = (int(track_frame_match.group(1)), int(track_frame_match.group(2)))  # Convert the captured track/frame strings to integers.
        if track_frame_pair not in selected_track_frame_pairs:  # Keep only explicitly requested pairs.
            continue  # Skip pairs that were not requested.
        if polarization_mode and polarization_mode_from_filename(granule_id) != polarization_mode:  # Apply the optional polarization filter.
            continue  # Skip products acquired in a different polarization mode.
        if granule_id in granule_urls:  # Same granule returned twice – nothing new to add.
            continue  # Skip duplicate granules.

        urls: list[str] = []  # URLs kept for this granule.
        seen_filenames: set[str] = set()  # Filenames already kept, used for de-duplication.
        for url in get_product_urls(product):  # Iterate over every URL ASF lists for this product.
            total_urls += 1  # Count every URL for the summary log.
            filename = filename_from_url(url)  # Extract the decoded filename from the URL.
            if not url.lower().startswith("https://") or not filename.startswith(granule_id):  # Keep only HTTPS links whose filename belongs to this granule.
                continue  # Not a granule file (e.g. the s3credentials endpoint).
            if PRIVATE_JPL_BUCKET in url and not include_private_files:  # Skip JPL-private files unless explicitly enabled.
                continue  # Move on to the next URL.
            if filename in seen_filenames:  # Skip duplicate filenames.
                continue  # Move on to the next URL.
            seen_filenames.add(filename)  # Remember this filename.
            urls.append(url)  # Keep this URL for the granule.

        has_main_hdf5 = any(  # Check that the main HDF5 product is among the URLs.
            filename_from_url(url).lower() in (f"{granule_id}.h5".lower(), f"{granule_id}.hdf5".lower())  # Compare each filename case-insensitively with <granule_id>.h5/.hdf5.
            for url in urls  # Iterate over the kept URLs.
        )  # End of any() check.
        if not has_main_hdf5:  # Without the main HDF5 the granule is unusable.
            logging.warning(f"Skipping {granule_id}: ASF lists no main HDF5 file for it.")  # Warn that the granule is being skipped.
            continue  # Skip this granule.
        granule_urls[granule_id] = urls  # Store the granule's URL list.

    file_count = sum(len(urls) for urls in granule_urls.values())  # Total number of files selected across all granules.
    logging.info(  # Log a summary of the selection.
        f"Selected {len(granule_urls)} granule(s) with {file_count} file(s) out of {total_urls} "  # Granule/file/URL counts.
        f"total URLs (polarization mode: {polarization_mode or 'ANY'})."  # Active polarization filter.
    )  # End of logging call.
    for granule_id, urls in granule_urls.items():  # Log each granule with its file count.
        logging.info(f"  {granule_id}: {len(urls)} file(s)")  # One indented line per granule.

    if not granule_urls:  # Nothing matched the filters.
        logging.warning("No matching NISAR granules found in search results. Exiting script.")  # Warn that there is nothing to download.
        sys.exit(0)  # Exit cleanly (status 0) since this is not an error.

    return granule_urls  # Return the per-granule URL mapping.

# --------------------------------------------------------------------------- #
# Download – resumable sequential downloads with retry and slow-link handling.
# --------------------------------------------------------------------------- #
def filename_from_url(url: str) -> str:  # Get the filename part of a URL.
    """Return the decoded final path component without query parameters."""
    return unquote(os.path.basename(urlparse(url).path))  # Parse the URL path, take its basename, and percent-decode it.


def content_range_total(content_range: str | None) -> int | None:  # Parse the total size from a Content-Range header.
    """Extract the complete object size from an HTTP Content-Range header."""
    total_text = (content_range or "").rpartition("/")[-1]  # Take the text after the final '/' (e.g. 'bytes 0-99/1000' -> '1000').
    return int(total_text) if total_text.isdigit() else None  # Return it as int, or None when unknown ('*').


def response_total_bytes(response: requests.Response) -> int | None:  # Determine the full remote file size of a response.
    """Return the full remote file size when the server provides it."""
    if response.status_code == requests.codes.partial_content:  # Partial (206) responses report the total in Content-Range.
        return content_range_total(response.headers.get("Content-Range"))  # Read the total from Content-Range.
    content_length = response.headers.get("Content-Length")  # Full (200) responses report the size in Content-Length.
    return int(content_length) if content_length and content_length.isdigit() else None  # Return it as int, or None when missing.


class SlowDownloadError(requests.ConnectionError):  # Custom error for persistently slow downloads (retried like a connection error).
    """Raised when a transfer remains below the configured useful speed."""


def download_single_file(url: str, output_directory: str, session: asf.ASFSession) -> None:  # Download one file with resume support.
    """Stream one file to a resumable .part file and finalize it atomically."""
    filename = filename_from_url(url)  # Filename taken from the URL.
    destination_path = os.path.join(output_directory, filename)  # Final destination path for the file.
    partial_path = f"{destination_path}.part"  # Temporary path used while downloading.
    starting_bytes = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0  # Bytes already downloaded by a previous attempt.
    headers = {"Range": f"bytes={starting_bytes}-"} if starting_bytes else {}  # Ask the server to resume from that byte, if any.

    with session.get(  # Open a streamed HTTP GET request.
        url,  # File URL to download.
        headers=headers,  # Optional Range header for resuming.
        stream=True,  # Stream the body instead of loading it into memory.
        timeout=(Config.DOWNLOAD_CONNECT_TIMEOUT, Config.DOWNLOAD_READ_TIMEOUT),  # (connect timeout, read timeout) in seconds.
    ) as response:  # Response is closed automatically when the block ends.
        if response.status_code == requests.codes.requested_range_not_satisfiable:  # 416: requested range is beyond the end of the file.
            total_bytes = content_range_total(response.headers.get("Content-Range"))  # Read the real file size from Content-Range.
            if total_bytes is not None and starting_bytes == total_bytes:  # The .part file is already complete.
                os.replace(partial_path, destination_path)  # Finalize it by renaming to the destination.
                return  # Nothing more to download.
            with open(partial_path, "wb"):  # Otherwise truncate the corrupt .part file.
                pass  # Empty block: opening in 'wb' mode truncates the file.
            raise IOError(  # Raise so the retry loop starts again from byte 0.
                f"Server rejected resume at byte {starting_bytes}; partial file was reset"  # Error message for the rejected resume.
            )  # End of raise statement.
        response.raise_for_status()  # Raise an exception for any other HTTP error status.

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content  # Append only when resuming and the server honoured the range (206).
        if append:  # Verify the resumed range starts at the expected byte.
            content_range = response.headers.get("Content-Range", "")  # Read the Content-Range header.
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)  # Extract the start byte of the returned range.
            if not match or int(match.group(1)) != starting_bytes:  # Range is missing or does not match our offset.
                with open(partial_path, "wb"):  # Truncate the .part file to avoid corruption.
                    pass  # Empty block: opening in 'wb' mode truncates the file.
                raise IOError(  # Raise so the retry loop restarts cleanly.
                    f"Unexpected Content-Range while resuming: {content_range!r}; partial file was reset"  # Error message for the mismatched range.
                )  # End of raise statement.
        elif starting_bytes:  # Server ignored the Range header and sent the full file (200).
            logging.warning("%s ignored its range request; restarting its partial download.", filename)  # Warn that the download restarts from scratch.
            starting_bytes = 0  # Reset the progress starting point.

        total_bytes = response_total_bytes(response)  # Full remote size, if known.
        if total_bytes is not None and starting_bytes > total_bytes:  # Local partial file larger than the remote object means corruption.
            with open(partial_path, "wb"):  # Truncate the .part file.
                pass  # Empty block: opening in 'wb' mode truncates the file.
            raise IOError(  # Raise so the retry loop restarts cleanly.
                f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes); partial file was reset"  # Error message for the oversized partial file.
            )  # End of raise statement.

        with open(partial_path, "ab" if append else "wb") as output_file, tqdm(  # Open the .part file (append or overwrite) and a progress bar together.
            total=total_bytes,  # Progress bar total = full file size (None if unknown).
            initial=starting_bytes,  # Start the bar at the already-downloaded byte count.
            unit="B",  # Display units in bytes.
            unit_scale=True,  # Scale units automatically (KB, MB, GB).
            unit_divisor=1024,  # Use 1024 as the scaling divisor.
            desc=filename[:70],  # Label the bar with the (truncated) filename.
            file=sys.stdout,  # Draw the bar on standard output.
            leave=True,  # Keep the finished bar visible.
        ) as progress_bar:  # End of with-statement setup.
            speed_window_started = time.monotonic()  # Start time of the current speed-measurement window.
            speed_window_bytes = 0  # Bytes received in the current speed-measurement window.
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):  # Read the response body in fixed-size chunks.
                if not chunk:  # Skip keep-alive empty chunks.
                    continue  # Move on to the next chunk.
                output_file.write(chunk)  # Write the chunk to the .part file.
                progress_bar.update(len(chunk))  # Advance the progress bar.
                speed_window_bytes += len(chunk)  # Add the chunk size to the speed window.

                elapsed = time.monotonic() - speed_window_started  # Seconds elapsed in the current speed window.
                transfer_complete = total_bytes is not None and progress_bar.n >= total_bytes  # True once all expected bytes have arrived.
                if (  # Start of slow-download check condition.
                    Config.DOWNLOAD_MIN_SPEED > 0  # Speed checking is enabled.
                    and elapsed >= Config.DOWNLOAD_SPEED_WINDOW  # A full measurement window has elapsed.
                    and not transfer_complete  # Do not flag the transfer once it has finished.
                ):  # End of condition.
                    average_speed = speed_window_bytes / elapsed  # Average bytes per second over the window.
                    if average_speed < Config.DOWNLOAD_MIN_SPEED:  # Throughput is below the configured minimum.
                        raise SlowDownloadError(  # Abort this attempt so it can be retried.
                            f"Sustained speed {average_speed / 1024:.1f} KiB/s is below "  # Report the measured speed in KiB/s.
                            f"the {Config.DOWNLOAD_MIN_SPEED / 1024:.1f} KiB/s limit"  # Report the configured limit in KiB/s.
                        )  # End of raise statement.
                    speed_window_started = time.monotonic()  # Start a new measurement window.
                    speed_window_bytes = 0  # Reset the window byte counter.

    downloaded_bytes = os.path.getsize(partial_path)  # Size of the .part file after the transfer ends.
    if total_bytes is not None and downloaded_bytes != total_bytes:  # Verify that the full file was received.
        raise IOError(f"Incomplete download: {downloaded_bytes} of {total_bytes} bytes received")  # Raise so the retry loop resumes the transfer.
    os.replace(partial_path, destination_path)  # Atomically rename the completed .part file to its final name.


def download_with_retries(  # Download a file, retrying failures with exponential backoff.
    url: str,  # File URL to download.
    output_directory: str,  # Folder to save the file in.
    session: asf.ASFSession,  # Authenticated session used for the request.
) -> str:  # Returns the downloaded filename.
    """Retry an interrupted/stalled transfer while retaining downloaded bytes."""
    filename = filename_from_url(url)  # Filename taken from the URL, used in messages.
    for attempt in range(1, Config.DOWNLOAD_MAX_ATTEMPTS + 1):  # Loop over the allowed number of attempts.
        try:  # Begin try block for one download attempt.
            download_single_file(url, output_directory, session)  # Attempt (or resume) the download.
            return filename  # Success: return the filename.
        except (requests.RequestException, OSError, ValueError) as error:  # Catch network, file-system and parsing errors.
            if attempt == Config.DOWNLOAD_MAX_ATTEMPTS:  # Last attempt failed: give up.
                raise RuntimeError(  # Raise a final error describing the failure.
                    f"{filename} failed after {attempt} attempts: {error}"  # Error message with the filename and attempt count.
                ) from error  # Chain the original exception for context.
            wait_seconds = min(  # Compute the backoff delay before the next attempt.
                Config.DOWNLOAD_MAX_BACKOFF,  # Upper bound on the delay.
                Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1)),  # Base delay doubled after each failed attempt.
            )  # End of min() call.
            logging.warning(  # Log the failure and the upcoming retry.
                "%s failed on attempt %d/%d: %s. Retrying from the partial file in %d seconds.",  # Log message template.
                filename,  # Filename for the message.
                attempt,  # Current attempt number.
                Config.DOWNLOAD_MAX_ATTEMPTS,  # Maximum number of attempts.
                error,  # The error that occurred.
                wait_seconds,  # Seconds to wait before retrying.
            )  # End of logging call.
            time.sleep(wait_seconds)  # Wait before the next attempt.

# --------------------------------------------------------------------------- #
# Paths – long-path support for the per-granule folders.
# --------------------------------------------------------------------------- #
def to_long_path(path: str) -> str:  # Convert a path to a Windows long-path form when needed.
    """Return a path usable beyond Windows' 260-character MAX_PATH limit.

    NISAR granule IDs are ~90 characters and appear in both the staging folder
    and the file names, so full paths easily exceed MAX_PATH on Windows.
    """
    absolute_path = os.path.abspath(path)  # Resolve the path to an absolute path.
    if os.name == "nt" and not absolute_path.startswith("\\\\?\\"):  # On Windows, add the long-path prefix if it is not already present.
        return "\\\\?\\" + absolute_path  # The prefix bypasses the 260-character MAX_PATH limit.
    return absolute_path  # Other platforms: return the absolute path unchanged.

# --------------------------------------------------------------------------- #
# Download – sequential per-granule download into one folder per granule.
# --------------------------------------------------------------------------- #
def download_granules_sequentially(  # Download every granule's files into its own folder.
    granule_urls: dict[str, list[str]],  # {granule_id: [HDF5 + metadata + visualization URLs]}.
    output_directory: str,  # Folder that holds one <granule_id>/ sub-folder per granule.
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:  # Returns nothing.
    """Download each granule's file bundle into ``<output_directory>/<granule_id>/``.

    Files are downloaded resumably, one progress bar per file.  Files already
    present are skipped, and a granule with failed files keeps whatever was
    downloaded so the next run resumes instead of starting over.
    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output folder if it does not exist.
    logging.info(f"Data download directory: {os.path.abspath(output_directory)}")  # Log the absolute download directory.
    logging.info(f"Starting sequential download of {len(granule_urls)} granule(s)...")  # Log how many granules will be processed.

    downloaded_granules = 0  # Counter of granules fully downloaded in this run.
    skipped_granules = 0  # Counter of granules already complete before this run.
    failed_granules = 0  # Counter of granules with at least one failed file.

    for granule_id, urls in granule_urls.items():  # Process each granule in turn.
        granule_directory = to_long_path(os.path.join(output_directory, granule_id))  # Per-granule folder path (long-path safe on Windows).
        if all(os.path.isfile(os.path.join(granule_directory, filename_from_url(url))) for url in urls):  # Check whether every file of this granule already exists.
            logging.info(f"All files already downloaded, skipping granule: {granule_id}")  # Log that the granule is already complete.
            skipped_granules += 1  # Count it as skipped.
            continue  # Move on to the next granule.

        os.makedirs(granule_directory, exist_ok=True)  # Create the granule folder if needed.
        logging.info(f"Downloading granule {granule_id} ({len(urls)} file(s))")  # Log the start of this granule's download.

        failed_files: list[str] = []  # Filenames that failed to download for this granule.
        for url in urls:  # Process each file of the granule.
            filename = filename_from_url(url)  # Filename taken from the URL.
            destination_path = os.path.join(granule_directory, filename)  # Final path of the file in the granule folder.

            if os.path.isfile(destination_path):  # File already downloaded in a previous run.
                logging.info(f"File already exists, skipping: {filename}")  # Log the skip.
                continue  # Move on to the next file.

            # Reuse a loose file left in the output folder by earlier versions of this script.
            legacy_path = to_long_path(os.path.join(output_directory, filename))  # Possible older location directly in the output folder.
            if os.path.isfile(legacy_path):  # Older copy exists.
                logging.info(f"Reusing previously downloaded file: {filename}")  # Log that the existing file is being reused.
                os.replace(legacy_path, destination_path)  # Move it into the granule folder.
                continue  # Move on to the next file.

            try:  # Begin try block for downloading this file.
                download_with_retries(url, granule_directory, session)  # Download the file with retries.
                logging.info(f"Successfully finished downloading file: {filename}")  # Log successful completion.
            except Exception as file_error:  # Catch any final download failure.
                failed_files.append(filename)  # Record the failed filename.
                logging.error(f"Failed to download {filename}: {file_error}")  # Log the failure details.

        if failed_files:  # At least one file of this granule failed.
            failed_granules += 1  # Count the granule as failed.
            logging.error(  # Log that the granule is incomplete.
                f"Granule {granule_id} incomplete: {len(failed_files)} file(s) failed. "  # Number of failed files.
                "Downloaded files are kept for the next run."  # Partial files remain for resuming next time.
            )  # End of logging call.
            continue  # Move on to the next granule.

        downloaded_granules += 1  # All files succeeded: count the granule as downloaded.
        logging.info(f"Granule complete: {os.path.abspath(os.path.join(output_directory, granule_id))}")  # Log the completed granule folder.

    logging.info(  # Log the overall download summary.
        "Download complete. Summary -> "  # Summary prefix.
        f"Downloaded: {downloaded_granules}, Already complete: {skipped_granules}, Failed: {failed_granules}"  # Downloaded / already complete / failed counts.
    )  # End of logging call.

# --------------------------------------------------------------------------- #
# Entry point – orchestrates the full workflow using Config settings.
# --------------------------------------------------------------------------- #
def main() -> None:  # Run the full search-and-download workflow using Config settings.
    """Run the full search-and-download workflow using Config settings."""
    setup_logging(Config.LOG_DIRECTORY)  # Set up logging before anything else runs.

    try:  # Begin try block for resolving the search date range.
        start_date, end_date = resolve_date_range(  # Compute the start/end dates from the configured mode.
            Config.DATE_MODE,  # Mode 1 (manual) or 2 (rolling).
            Config.START_DATE,  # Manual start date.
            Config.END_DATE,  # Manual end date.
            Config.DATE_LOOKBACK_DAYS,  # Lookback days for rolling mode.
        )  # End of function call.
    except ValueError as date_error:  # Catch invalid date configuration.
        logging.error("Invalid date configuration: %s", date_error)  # Log the date configuration error.
        sys.exit(1)  # Exit with failure.

    session = authenticate_earthdata(Config.EARTHDATA_USERNAME, Config.EARTHDATA_PASSWORD)  # Log in to Earthdata.

    try:  # Begin try block for validating the polarization mode.
        normalize_polarization_mode(Config.POLARIZATION_MODE)  # Fail fast on an invalid polarization mode.
    except ValueError as mode_error:  # Catch an invalid polarization mode.
        logging.error("Invalid polarization configuration: %s", mode_error)  # Log the polarization configuration error.
        sys.exit(1)  # Exit with failure.

    aoi_wkt = load_aoi_wkt_from_shapefile(Config.AOI_SHAPEFILE)  # Derive the WKT search geometry from the AOI shapefile.
    selected_track_frame_pairs, granule_patterns = prepare_track_frame_filters(  # Validate Track/Frame pairs and build filename patterns.
        Config.TRACK_FRAME_PAIRS  # Configured Track/Frame pairs.
    )  # End of function call.

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,  # Spatial filter geometry.
        start_date=start_date,  # Search start date.
        end_date=end_date,  # Search end date.
        product_level=Config.PRODUCT_LEVEL,  # Processing level (e.g. GSLC).
        frame_coverage=Config.FRAME_COVERAGE,  # Full-frame coverage only.
        granule_patterns=granule_patterns,  # Track/Frame filename patterns.
        max_results=Config.MAX_RESULTS,  # Maximum number of results.
    )  # End of function call.

    granule_urls = group_granule_urls(  # Every HDF5, metadata and visualization file per granule.
        results,  # Search results to group.
        selected_track_frame_pairs,  # Track/Frame pairs to keep.
        Config.POLARIZATION_MODE,  # Polarization mode filter.
        Config.INCLUDE_PRIVATE_FILES,  # Whether to include JPL-private files.
    )  # End of function call.

    download_granules_sequentially(  # Download each granule's files into its own folder.
        granule_urls,  # Per-granule URL mapping.
        Config.OUTPUT_DIRECTORY,  # Root output folder.
        session,  # Authenticated Earthdata session.
    )  # End of function call.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
