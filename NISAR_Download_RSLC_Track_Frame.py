"""
NISAR_Download_RSLC_Track_Frame.py

Search NASA's ASF (Alaska Satellite Facility) catalog for NISAR L1 RSLC
granules within a given area of interest, date range and Track/Frame
selection, and download *every* file that belongs to each granule – not only
the main HDF5 product.

For each granule ASF publishes a bundle of companion files, for example:
    <granule>.h5                  main RSLC product
    <granule>_QA_STATS.h5         QA statistics
    <granule>_QA_REPORT.pdf       QA report
    <granule>_QA_SUMMARY.csv      QA summary
    <granule>.h5.iso.xml          ISO metadata
    <granule>.rc.yaml             run configuration
    <granule>_*.kml               footprints (LATLON / NATIVE / per-pol)
    <granule>_*.png               browse images and thumbnail
    <granule>.met.json / .context.json / .dataset.json / .h5.log
                                  JPL processing metadata

All files of one granule are saved together in a folder named after the
granule following the NISAR naming convention, e.g.:

    NISAR_Product_RSLC/
        NISAR_L1_PR_RSLC_031_011_A_009_4005_DHDH_A_20260918T224526_20260918T224604_P05023_F_F_J_001/
            NISAR_L1_PR_RSLC_031_011_A_009_..._001.h5
            NISAR_L1_PR_RSLC_031_011_A_009_..._001_QA_STATS.h5
            ...

Each file download shows its own byte-level progress bar (current bytes / total bytes),
rather than a single progress bar tracking file count.

Requirements:
    pip install asf_search tqdm requests geopandas shapely python-dotenv

Credentials:
    Copy .env.example to .env and set EARTHDATA_USERNAME and
    EARTHDATA_PASSWORD there before running the script. ".env" is git-ignored.

Usage:
    python NISAR_Download_RSLC_Track_Frame.py
"""  # End of module‑level docstring – describes the whole script.
# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).
import logging  # Standard logging module for recording execution steps, warnings, and errors.
import re  # Regular-expression support for confirming Track/Frame values in returned filenames.
import time  # Sleep between download retries when the connection drops.
from datetime import datetime, timedelta  # Date/time tools for configured and rolling search windows.

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
    OUTPUT_DIRECTORY = "NISAR_Product_RSLC"  # Parent folder; each granule gets its own sub-folder here.

    # Area of interest, as a path to a shapefile (.shp). All features in the
    # file are dissolved into a single geometry and reprojected to WGS84
    # (EPSG:4326) automatically before being sent to the ASF search API.

    # Directory where this .py file lives
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Build the full path relative to that directory
    AOI_SHAPEFILE = os.path.join(script_dir,
                                 "Administrative_Boundary_dir",
                                 "Administrative_Boundary.shp")

    # Mode 1 uses the manually configured dates below. Mode 2 ignores them and
    # searches from DATE_LOOKBACK_DAYS ago through the current date and time.
    DATE_MODE = 1
    START_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    END_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    DATE_LOOKBACK_DAYS = 10

    PRODUCT_LEVEL = "RSLC"  # NISAR processing level to filter results by (L1 Range-Doppler SLC).
    FRAME_COVERAGE = "FULL"  # Exclude NISAR products that cover only a partial frame.

    # NISAR Track/Frame pairs to download.  Add every required pair here as
    # (track, frame); for example, (105, 78) means Track 105, Frame 078.
    #
    # A pair is downloaded only when its full-frame product footprint
    # intersects AOI_SHAPEFILE; partial-frame products are excluded.
    # Leave no pairs configured only if you want the script to stop before
    # searching, rather than accidentally downloading every AOI result.
    TRACK_FRAME_PAIRS: list[tuple[int, int]] = [
        (4, 82),
    ]

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk, used for streaming downloads and progress updates.

    # Resilience for slow / unstable connections.  An interrupted file is kept
    # as "<name>.part" and resumed from where it stopped (HTTP Range request),
    # both on automatic retries and on the next run of the script.
    DOWNLOAD_CONNECT_TIMEOUT = 30  # Seconds to wait for the server to accept the connection.
    DOWNLOAD_READ_TIMEOUT = 120  # Seconds without receiving any data before the connection is treated as stalled.
    # Retries allowed in a row *without any progress*; any attempt that receives
    # new bytes resets the counter, so a long file on a flaky link keeps going.
    DOWNLOAD_MAX_RETRIES = 10
    DOWNLOAD_RETRY_BASE_DELAY = 10  # Seconds before the first retry; doubles on each consecutive failure.
    DOWNLOAD_RETRY_MAX_DELAY = 300  # Upper limit for the wait between retries (5 minutes).

    # Some companion files (.met.json, .context.json, .dataset.json, .h5.log)
    # are stored in ASF's "NISAR-JPL-PRIVATE-DATA" bucket and may be refused
    # (HTTP 401/403) for ordinary Earthdata accounts.  When True they are
    # still attempted, and an access refusal is logged as a warning rather
    # than counted as a failed download.  Set to False to skip them entirely.
    INCLUDE_PRIVATE_JPL_FILES = True


def resolve_date_range(
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
    )

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
    log_filename = f"nisar_rslc_download_{timestamp}.log"  # Build a dynamic, timestamped log filename.
    log_filepath = os.path.join(log_directory, log_filename)  # Construct the full path for the log file.

    logging.basicConfig(  # Initialize the global logging configuration.
        level=logging.INFO,  # Set minimum logging threshold to INFO level.
        format="%(asctime)s [%(levelname)s] %(message)s",  # Set timestamped format for log output lines.
        handlers=[  # Specify list of active log message destinations.
            logging.FileHandler(log_filepath),  # Handler 1: save log output to the log file.
            logging.StreamHandler(sys.stdout),  # Handler 2: display log output on standard stdout,
        ],
    )

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
    if not username or not password:
        logging.error(
            "NASA Earthdata credentials are missing. Set EARTHDATA_USERNAME "
            "and EARTHDATA_PASSWORD in the .env file (see .env.example)."
        )
        sys.exit(1)

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
                "Shapefile has no defined CRS; assuming it is already WGS84 (EPSG:4326)."
            )
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
def prepare_track_frame_filters(
    track_frame_pairs: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[str]]:
    """Validate Track/Frame pairs and create NISAR filename search patterns.

    ASF documents that NISAR currently lacks searchable Track/Frame metadata.
    NISAR products do, however, encode them in their granule names as
    ``..._<track>_<direction>_<frame>_...``.  ``granule_list`` patterns are
    therefore used for the Track/Frame part of the search, while
    ``intersectsWith`` remains the spatial AOI filter.
    """
    if not track_frame_pairs:
        logging.error(
            "No TRACK_FRAME_PAIRS configured. Add one or more (track, frame) "
            "pairs in Config before running the script."
        )
        sys.exit(1)

    selected_pairs: set[tuple[int, int]] = set()
    for pair in track_frame_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            logging.error(
                "Each TRACK_FRAME_PAIRS entry must be a two-item tuple "
                "such as (105, 78)."
            )
            sys.exit(1)

        track, frame = pair
        if (
            isinstance(track, bool)
            or isinstance(frame, bool)
            or not isinstance(track, int)
            or not isinstance(frame, int)
            or track < 0
            or frame < 0
        ):
            logging.error(
                f"Invalid Track/Frame pair {pair!r}. Both values must be non-negative integers."
            )
            sys.exit(1)
        selected_pairs.add((track, frame))

    # The `?` is the ascending/descending direction character between Track
    # and Frame.  The leading `*` accommodates the NISAR product/version
    # fields that precede the Track in the granule name.
    granule_patterns = [
        f"NISAR_*{track:03d}_?_{frame:03d}_*"
        for track, frame in sorted(selected_pairs)
    ]
    return selected_pairs, granule_patterns


def search_nisar_granules(  # Query the ASF catalog for NISAR granules matching the given filters.
    aoi_wkt: str,  # Area of interest as a WKT geometry string.
    start_date: datetime,  # Earliest acquisition date to include.
    end_date: datetime,  # Latest acquisition date to include.
    product_level: str,  # NISAR processing level to filter on (e.g. "RSLC").
    granule_patterns: list[str],  # NISAR filename patterns for selected Track/Frame pairs.
    max_results: int,  # Maximum number of granules to return.
    frame_coverage: str = "FULL",  # NISAR coverage to include ("FULL" excludes partial scenes).

) -> asf.ASFSearchResults:  # The raw ASF search results object.
    """Query the ASF catalog for NISAR granules matching the given filters.

    Args:
        aoi_wkt: Area of interest as a WKT geometry string.
        start_date: Earliest acquisition date to include.
        end_date: Latest acquisition date to include.
        product_level: NISAR processing level to filter on (e.g. "RSLC").
        granule_patterns: NISAR filename patterns representing the selected
            Track/Frame pairs.
        max_results: Maximum number of granules to return.
        frame_coverage: NISAR frame coverage to include.

    Returns:
        The raw ASF search results object.

    Exits:
        If the search request fails.
    """
    logging.info(f"Area of Interest (AOI WKT): {aoi_wkt}")  # Log the specified spatial coverage WKT boundary.
    logging.info(f"Search date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")  # Log the search time range.
    logging.info(f"Target Processing Level: {product_level}")  # Log the selected target product level.
    logging.info(f"Target Frame Coverage: {frame_coverage}")  # FULL prevents partial-frame products from being returned.
    logging.info(
        "Selected Track/Frame granule pattern(s): " + ", ".join(granule_patterns)
    )

    search_options = asf.ASFSearchOptions(  # Initialize the ASF search configuration object.
        dataset=["NISAR"],  # Filter search results to the NISAR dataset platform.
        # Keeps any product whose footprint has *any* overlap with the AOI;
        # full containment is not required.
        intersectsWith=aoi_wkt,
        # NISAR Track/Frame is encoded in its filename. This works with
        # NISAR even where CMR has no searchable Track/Frame metadata.
        granule_list=granule_patterns,
        start=start_date,  # Filter granules acquired on or after the start date.
        end=end_date,  # Filter granules acquired on or before the end date.
        processingLevel=[product_level],  # Filter granules by the specified product level.
        frameCoverage=frame_coverage,  # Retain full-frame products and exclude partial scenes.
        maxResults=max_results,  # Limit the maximum number of search results retrieved.
    )

    logging.info("Querying ASF search API for matching NISAR datasets...")  # Log the start of the query.
    try:  # Begin try block for the API search request.
        results = asf.search(opts=search_options)  # Execute the catalog search query against the ASF API.
        logging.info(f"Search completed. Found {len(results)} matching granules.")  # Log the total hit count.
        return results  # Return the search results object.
    except Exception as search_error:  # Intercept any search query errors.
        logging.error(f"Search query failed: {search_error}")  # Log search query failure details.
        sys.exit(1)  # Exit script execution with a failure status code.

# --------------------------------------------------------------------------- #
# Group – collects every companion file URL per granule, keyed by granule ID.
# --------------------------------------------------------------------------- #
# Captures NISAR's ..._<cycle>_<track>_<direction>_<frame>_... granule fields,
# for both L1 (RSLC/RIFG/...) and L2 (GSLC/GCOV/...) products.
TRACK_FRAME_PATTERN = re.compile(r"^NISAR_L\d_[^_]+_[^_]+_\d+_(\d+)_[AD]_(\d+)_")

# Bucket path segment used by ASF for JPL processing metadata that may be
# access-restricted (see Config.INCLUDE_PRIVATE_JPL_FILES).
PRIVATE_JPL_BUCKET = "/NISAR-JPL-PRIVATE-DATA/"


def to_long_path(path: str) -> str:
    """Return a path usable beyond Windows' 260-character MAX_PATH limit.

    NISAR granule IDs are ~90 characters and appear in both the folder and
    the file name, so full paths easily exceed MAX_PATH.  On Windows the
    ``\\\\?\\`` prefix lifts that limit; elsewhere the path is returned as-is.
    """
    absolute_path = os.path.abspath(path)
    if os.name == "nt" and not absolute_path.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute_path
    return absolute_path


def get_product_urls(product: asf.ASFProduct) -> list[str]:
    """Return every download URL ASF lists for one product.

    Uses ``ASFProduct.find_urls()`` where available (asf_search >= 8), which
    covers the main file, ``additionalUrls`` and browse images.  Falls back
    to reading the product properties directly on older asf_search versions.
    """
    if hasattr(product, "find_urls"):
        return list(product.find_urls())

    properties = product.properties
    urls = [properties.get("url")]
    urls.extend(properties.get("additionalUrls") or [])
    browse = properties.get("browse") or []
    urls.extend([browse] if isinstance(browse, str) else browse)
    return [url for url in urls if url]


def group_granule_urls(
    results: asf.ASFSearchResults,
    selected_track_frame_pairs: set[tuple[int, int]],
    include_private_files: bool,
) -> dict[str, list[str]]:
    """Group the search results into {granule_id: [file URLs]}.

    Every file that belongs to a granule is kept (HDF5, QA, KML, PNG, XML,
    YAML, JSON, PDF, CSV, ...).  Only non-file links such as ``s3credentials``
    are dropped, i.e. any URL whose filename does not start with the granule
    ID.

    Args:
        results: Search results returned by search_nisar_granules().
        selected_track_frame_pairs: Exact (track, frame) pairs to retain.
        include_private_files: Whether to keep files from the
            NISAR-JPL-PRIVATE-DATA bucket.

    Returns:
        Ordered mapping of granule ID (NISAR naming convention, no extension)
        to the de-duplicated list of that granule's download URLs.

    Exits:
        Gracefully (status 0) if no matching granules are found.
    """
    granule_urls: dict[str, list[str]] = {}
    total_urls = 0

    for product in results:
        granule_id = product.properties.get("sceneName") or product.properties.get("fileID")
        if not granule_id:
            logging.warning("Skipping a search result without a sceneName/fileID.")
            continue
        granule_id = os.path.splitext(granule_id)[0] if granule_id.lower().endswith(".h5") else granule_id

        # This local check is deliberately kept in addition to the server-side
        # `granule_list` filter, so only exact requested pairs are downloaded.
        track_frame_match = TRACK_FRAME_PATTERN.match(granule_id)
        if track_frame_match is None:
            logging.warning(f"Skipping granule with an unrecognised NISAR name: {granule_id}")
            continue
        track_frame_pair = (int(track_frame_match.group(1)), int(track_frame_match.group(2)))
        if track_frame_pair not in selected_track_frame_pairs:
            continue

        if granule_id in granule_urls:  # Same granule returned twice – nothing new to add.
            continue

        urls: list[str] = []
        seen_filenames: set[str] = set()
        for url in get_product_urls(product):
            total_urls += 1
            filename = url.split("/")[-1]
            if not url.lower().startswith("https://") or not filename.startswith(granule_id):
                continue  # Not a granule file (e.g. the s3credentials endpoint).
            if PRIVATE_JPL_BUCKET in url and not include_private_files:
                continue
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            urls.append(url)

        if urls:
            granule_urls[granule_id] = urls

    file_count = sum(len(urls) for urls in granule_urls.values())
    logging.info(
        f"Selected {len(granule_urls)} granule(s) with {file_count} file(s) "
        f"out of {total_urls} total URLs."
    )
    for granule_id, urls in granule_urls.items():
        logging.info(f"  {granule_id}: {len(urls)} file(s)")

    if not granule_urls:
        logging.warning("No matching NISAR granules found in search results. Exiting script.")
        sys.exit(0)

    return granule_urls

# --------------------------------------------------------------------------- #
# Download – downloads a single file, showing a byte‑level tqdm progress bar.
# --------------------------------------------------------------------------- #
class IncompleteDownloadError(IOError):
    """The transfer ended before the whole file arrived; safe to resume/retry."""


# Server-side conditions that are usually temporary and worth retrying.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Network failures that are worth retrying (dropped / stalled / reset connections).
RETRYABLE_NETWORK_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    IncompleteDownloadError,
)

# "Content-Range: bytes <start>-<end>/<total>" (total may be "*" if unknown).
CONTENT_RANGE_PATTERN = re.compile(r"bytes\s+(\d+)-\d+/(\d+|\*)")
# "Content-Range: bytes */<total>" as sent with HTTP 416.
UNSATISFIED_RANGE_PATTERN = re.compile(r"bytes\s+\*/(\d+)")


def get_file_size(path: str) -> int:
    """Return the size of ``path`` in bytes, or 0 if it does not exist."""
    return os.path.getsize(path) if os.path.isfile(path) else 0


def download_attempt(
    url: str,
    destination_path: str,
    partial_path: str,
    filename: str,
    session: asf.ASFSession,
) -> None:
    """Make one attempt to download ``url``, resuming any existing ``.part`` file.

    Raises:
        requests.HTTPError: If the server returns a non-success status code.
        IncompleteDownloadError: If the transfer ended early or could not be resumed.
        requests.ConnectionError / requests.Timeout: If the connection dropped or stalled.
    """
    resume_from = get_file_size(partial_path)
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    timeout = (Config.DOWNLOAD_CONNECT_TIMEOUT, Config.DOWNLOAD_READ_TIMEOUT)

    # Streamed GET so the body isn't loaded all at once; the timeout makes a
    # stalled connection raise instead of hanging forever.
    with session.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if response.status_code == 416 and resume_from:
            # Requested offset is past the end: either the .part already holds the
            # whole file (finished but not renamed) or it is invalid.
            match = UNSATISFIED_RANGE_PATTERN.match(response.headers.get("Content-Range", ""))
            if match and int(match.group(1)) == resume_from:
                os.replace(partial_path, destination_path)
                return
            os.remove(partial_path)
            raise IncompleteDownloadError("Server rejected the resume offset; restarting from the beginning.")

        response.raise_for_status()  # Raise an exception if the server returned an error status code.

        if response.status_code == 206:  # Partial content – the server honoured the resume request.
            match = CONTENT_RANGE_PATTERN.match(response.headers.get("Content-Range", ""))
            if match is None or int(match.group(1)) != resume_from:
                os.remove(partial_path)
                raise IncompleteDownloadError("Server returned an unexpected byte range; restarting from the beginning.")
            total_bytes = int(match.group(2)) if match.group(2) != "*" else 0
            file_mode = "ab"  # Append to what is already on disk.
            logging.info(f"Resuming {filename} from {resume_from / 1024**2:,.1f} MB.")
        else:  # Plain 200 – fresh download, or the server ignored the Range header.
            resume_from = 0
            total_bytes = int(response.headers.get("Content-Length", 0))
            file_mode = "wb"

        received_bytes = resume_from
        with open(partial_path, file_mode) as output_file, tqdm(  # Open the temporary file and a progress bar together.
            total=total_bytes or None,  # Progress bar's total is the file's expected size in bytes.
            initial=resume_from,  # Start the bar at the already-downloaded amount when resuming.
            unit="B",  # Display units as bytes.
            unit_scale=True,  # Auto-scale bytes to KB/MB/GB for readability.
            unit_divisor=1024,  # Use 1024 as the scaling divisor (binary units).
            desc=filename,  # Show the filename as the progress bar's label.
            file=sys.stdout,  # Render the progress bar to standard output.
            leave=True,  # Keep the completed bar visible after the file finishes.
        ) as progress_bar:  # Progress bar context manager.
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):  # Stream the file in fixed-size chunks.
                if chunk:  # Skip any empty keep-alive chunks.
                    output_file.write(chunk)  # Write this chunk to disk.
                    received_bytes += len(chunk)
                    progress_bar.update(len(chunk))  # Advance the progress bar by the chunk's byte size.

    if total_bytes and received_bytes != total_bytes:
        raise IncompleteDownloadError(f"Incomplete download: received {received_bytes} of {total_bytes} bytes.")

    os.replace(partial_path, destination_path)  # Promote the completed file to its final name.


def download_single_file(url: str, destination_path: str, session: asf.ASFSession) -> None:  # Download one file, showing a byte-level tqdm progress bar for it.
    """Download one file, resuming and retrying through connection problems.

    Streams the response in chunks rather than loading it all into memory,
    and updates the progress bar as each chunk arrives so the bar reflects
    real download progress (not just file count).

    The data is first written to ``<destination_path>.part`` and only renamed
    to its final name once the transfer completes, so an interrupted
    download is never mistaken for a finished file on the next run.  When the
    connection drops or stalls, the download is retried after a back-off delay
    and continues from the end of the ``.part`` file instead of starting over;
    a ``.part`` file left by an earlier run is resumed the same way.

    Args:
        url: Direct download URL for the file.
        destination_path: Full local path the file is saved to.
        session: Authenticated ASFSession (subclasses requests.Session, so it can be used directly for streamed HTTP GETs).

    Raises:
        requests.HTTPError: If the server returns a non-retryable error status,
            or a retryable one persists past DOWNLOAD_MAX_RETRIES.
        IOError / requests.RequestException: If the connection keeps failing
            without progress for DOWNLOAD_MAX_RETRIES consecutive attempts.
    """
    filename = os.path.basename(destination_path)
    destination_path = to_long_path(destination_path)
    partial_path = destination_path + ".part"

    failures_without_progress = 0
    while True:
        bytes_before = get_file_size(partial_path)
        try:
            download_attempt(url, destination_path, partial_path, filename, session)
            return
        except requests.HTTPError as http_error:
            status_code = http_error.response.status_code if http_error.response is not None else None
            if status_code not in RETRYABLE_STATUS_CODES:
                raise  # e.g. 401/403/404 – retrying will not help.
            last_error: Exception = http_error
        except RETRYABLE_NETWORK_ERRORS as network_error:
            last_error = network_error

        # Any new bytes on disk mean the link is working, just unreliable –
        # keep going without using up the retry budget.
        if get_file_size(partial_path) > bytes_before:
            failures_without_progress = 0
        else:
            failures_without_progress += 1
            if failures_without_progress > Config.DOWNLOAD_MAX_RETRIES:
                logging.error(
                    f"Giving up on {filename} after {Config.DOWNLOAD_MAX_RETRIES} retries without progress. "
                    f"Partial data is kept and will resume on the next run."
                )
                raise last_error

        delay = min(
            Config.DOWNLOAD_RETRY_BASE_DELAY * 2 ** max(failures_without_progress - 1, 0),
            Config.DOWNLOAD_RETRY_MAX_DELAY,
        )
        logging.warning(
            f"Download interrupted for {filename} ({last_error}). "
            f"{get_file_size(partial_path) / 1024**2:,.1f} MB saved; retrying in {delay} s "
            f"(retry {failures_without_progress}/{Config.DOWNLOAD_MAX_RETRIES} without progress)."
        )
        time.sleep(delay)

# --------------------------------------------------------------------------- #
# Download – sequential download of every granule's files into its own folder.
# --------------------------------------------------------------------------- #
def download_granules_sequentially(  # Download each granule's file bundle into a folder named after the granule.
    granule_urls: dict[str, list[str]],  # Granule ID -> URLs of its files.
    output_directory: str,  # Parent directory; one sub-folder per granule is created inside it.
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:
    """Download every granule's files, one granule (and one file) at a time.

    Args:
        granule_urls: Mapping from granule ID to the URLs of its files.
        output_directory: Parent directory; each granule's files are saved in
            ``<output_directory>/<granule_id>/``.
        session: Authenticated ASFSession used for the HTTP requests.
    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output data directory safely if it doesn't exist.
    logging.info(f"Data download directory: {os.path.abspath(output_directory)}")  # Log its absolute path.

    total_files = sum(len(urls) for urls in granule_urls.values())
    logging.info(
        f"Starting sequential download of {len(granule_urls)} granule(s), {total_files} file(s)..."
    )

    successful_downloads = 0  # Files downloaded now or already present.
    failed_downloads = 0  # Files that failed for any reason other than access restriction.
    restricted_downloads = 0  # Private JPL files the account is not permitted to read.

    for granule_index, (granule_id, urls) in enumerate(granule_urls.items(), start=1):
        granule_directory = os.path.join(output_directory, granule_id)
        os.makedirs(to_long_path(granule_directory), exist_ok=True)
        logging.info(f"[{granule_index}/{len(granule_urls)}] Granule folder: {granule_directory}")

        for url in urls:  # Iterate through the URLs one at a time (sequential, not parallel).
            filename = url.split("/")[-1]  # Extract the filename for logging purposes.
            destination_path = os.path.join(granule_directory, filename)  # Full local path where this file is saved.

            # Skip the file if it already exists locally – avoids re‑downloading.
            if os.path.isfile(to_long_path(destination_path)):
                logging.info(f"File already exists, skipping: {filename}")
                successful_downloads += 1  # Count it as a successful (already‑present) download.
                continue  # Move on to the next URL.

            logging.info(f"Starting download for file: {filename}")  # Log the start of this file's download.
            try:  # Begin try block for a single file download.
                download_single_file(url, destination_path, session)  # Download the file with its own progress bar.
                successful_downloads += 1  # Increment the success counter on completion.
                logging.info(f"Successfully finished downloading file: {filename}")  # Log successful completion.
            except requests.HTTPError as http_error:
                status_code = http_error.response.status_code if http_error.response is not None else None
                if PRIVATE_JPL_BUCKET in url and status_code in (401, 403):
                    restricted_downloads += 1
                    logging.warning(f"Access restricted (HTTP {status_code}), skipping private JPL file: {filename}")
                else:
                    failed_downloads += 1
                    logging.error(f"Failed to download {filename}: {http_error}")
            except Exception as file_error:  # Intercept any error for this specific file.
                failed_downloads += 1  # Increment the failure counter.
                logging.error(f"Failed to download {filename}: {file_error}")  # Log the detailed error message.

    logging.info(  # Log the overall download summary.
        f"Download complete. Summary -> Successful: {successful_downloads}, "
        f"Failed: {failed_downloads}, Access restricted: {restricted_downloads}"
    )

# --------------------------------------------------------------------------- #
# Entry point – orchestrates the full workflow using Config settings.
# --------------------------------------------------------------------------- #
def main() -> None:  # Run the full search-and-download workflow using Config settings.
    """Run the full search-and-download workflow using Config settings."""
    setup_logging(Config.LOG_DIRECTORY)  # Set up logging before anything else runs.

    try:
        start_date, end_date = resolve_date_range(
            Config.DATE_MODE,
            Config.START_DATE,
            Config.END_DATE,
            Config.DATE_LOOKBACK_DAYS,
        )
    except ValueError as date_error:
        logging.error("Invalid date configuration: %s", date_error)
        sys.exit(1)

    session = authenticate_earthdata(Config.EARTHDATA_USERNAME, Config.EARTHDATA_PASSWORD)  # Log in to Earthdata.

    aoi_wkt = load_aoi_wkt_from_shapefile(Config.AOI_SHAPEFILE)  # Derive the WKT search geometry from the AOI shapefile.
    selected_track_frame_pairs, granule_patterns = prepare_track_frame_filters(
        Config.TRACK_FRAME_PAIRS
    )

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,
        start_date=start_date,
        end_date=end_date,
        product_level=Config.PRODUCT_LEVEL,
        frame_coverage=Config.FRAME_COVERAGE,
        granule_patterns=granule_patterns,
        max_results=Config.MAX_RESULTS,
    )

    granule_urls = group_granule_urls(  # Collect every file of each exact Track/Frame granule.
        results, selected_track_frame_pairs, Config.INCLUDE_PRIVATE_JPL_FILES
    )

    download_granules_sequentially(granule_urls, Config.OUTPUT_DIRECTORY, session)  # Download each granule in turn.

    logging.info("NISAR RSLC search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
