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
                             "Thailand_Admin",
                             "tha_admbnda_adm1_rtsd_20190221.shp")  # Path to the shapefile defining the area of interest (AOI).

    # Mode 1 uses the manually configured dates below. Mode 2 ignores them and
    # searches from DATE_LOOKBACK_DAYS ago through the current date and time.
    DATE_MODE = 1
    START_DATE = datetime.strptime("2026-11-05", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    END_DATE = datetime.strptime("2026-11-06", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    DATE_LOOKBACK_DAYS = 10

    PRODUCT_LEVEL = "GSLC"  # NISAR processing level to filter results by.
    FRAME_COVERAGE = "FULL"  # Exclude NISAR products that cover only a partial frame.
    POLARIZATION_MODE = "DH"  # "SH" (single-pol H), "DH" (dual-pol H: HH+HV), or None to keep every mode.

    # NISAR Track/Frame pairs to download.  Add every required pair here as
    # (track, frame); for example, (105, 78) means Track 105, Frame 078.
    # A pair is downloaded only when its full-frame product footprint
    # intersects AOI_SHAPEFILE; partial-frame products are excluded.
    # Leave no pairs configured only if you want the script to stop before
    # searching, rather than accidentally downloading every AOI result.
    TRACK_FRAME_PAIRS: list[tuple[int, int]] = [
        (4, 81), (4, 82), (11, 8), (11, 9), (11, 10)
    ]

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 256 * 1024  # 256 KB chunks keep progress and slow-link checks responsive.
    DOWNLOAD_CONNECT_TIMEOUT = 30  # Seconds allowed to establish an HTTP connection.
    DOWNLOAD_READ_TIMEOUT = 180  # Seconds allowed without receiving any download data.
    DOWNLOAD_MIN_SPEED = 16 * 1024  # Retry when sustained throughput falls below 16 KiB/s; set to 0 to disable.
    DOWNLOAD_SPEED_WINDOW = 300  # Seconds over which sustained low throughput is measured.
    DOWNLOAD_MAX_ATTEMPTS = 8  # Initial attempt plus retries for interrupted or persistently slow downloads.
    DOWNLOAD_RETRY_BACKOFF = 5  # Base seconds between retries; doubles after each failure.
    DOWNLOAD_MAX_BACKOFF = 120  # Cap retry delays so recovery does not pause for an excessive period.


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
    log_filename = f"nisar_search_download_{timestamp}.log"  # Build a dynamic, timestamped log filename.
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
    logging.info(
        "Selected Track/Frame granule pattern(s): " + ", ".join(granule_patterns)
    )

    search_options = asf.ASFSearchOptions(  # Initialize the ASF search configuration object.
        dataset=["NISAR"],  # Filter search results to the NISAR dataset platform.
        # Spatially select frames that overlap the AOI. frameCoverage below
        # independently rejects NISAR products containing only part of a frame.
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
# Filter – groups every file (HDF5, metadata, visualizations) per granule.
# --------------------------------------------------------------------------- #
def normalize_polarization_mode(polarization_mode: str | None) -> str | None:
    """Return an upper-case polarization mode code, or ``None`` for no filtering.

    Args:
        polarization_mode: ``"SH"`` (single-pol H) or ``"DH"`` (dual-pol H); ``None``/``"ANY"`` keeps every mode.

    Raises:
        ValueError: If the value is not SH, DH, ANY, or None.
    """
    if polarization_mode is None or str(polarization_mode).strip().upper() in ("", "ANY"):
        return None  # No polarization filtering requested.
    mode = str(polarization_mode).strip().upper()  # Normalize case and whitespace.
    if mode not in ("SH", "DH"):
        raise ValueError("POLARIZATION_MODE must be 'SH', 'DH', or None.")  # Reject unsupported modes early.
    return mode


def polarization_mode_from_filename(filename: str) -> str | None:
    """Extract the frequency-A polarization mode (e.g. ``DH``) from a NISAR filename.

    NISAR names embed ``_<bandwidth>_<modeA><modeB>_`` such as ``_2005_DHDH_``;
    the first two letters describe the frequency-A polarization mode.
    """
    match = re.search(r"_\d{4}_([A-Z]{2})([A-Z]{2})_", filename)  # Locate the 4-letter mode token after the bandwidth code.
    return match.group(1) if match else None  # None when the name has no recognizable mode token.


# Capture NISAR's ..._<track>_<direction>_<frame>_... granule-name fields.
TRACK_FRAME_PATTERN = re.compile(r"^NISAR_L2_[^_]+_[^_]+_\d+_(\d+)_[AD]_(\d+)_")

# ASF lists some files from a JPL-internal bucket that ordinary Earthdata accounts cannot access.
PRIVATE_JPL_BUCKET = "/NISAR-JPL-PRIVATE-DATA/"


def get_product_urls(product: asf.ASFProduct) -> list[str]:
    """Return every HTTPS URL ASF lists for one product.

    Covers the main HDF5 file plus ``additionalUrls`` (QA_STATS, ISO XML,
    KML, JSON/YAML metadata, QA reports) and browse images (PNG).
    """
    if hasattr(product, "find_urls"):
        urls = list(product.find_urls())
    else:  # Older asf_search versions without ASFProduct.find_urls().
        properties = product.properties
        urls = [properties.get("url")]
        urls.extend(properties.get("additionalUrls") or [])
        browse = properties.get("browse") or []
        urls.extend([browse] if isinstance(browse, str) else browse)
    return [url for url in urls if url]


def group_granule_urls(
    results: asf.ASFSearchResults,
    selected_track_frame_pairs: set[tuple[int, int]],
    polarization_mode: str | None = None,
    include_private_files: bool = False,
) -> dict[str, list[str]]:
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
    granule_urls: dict[str, list[str]] = {}
    total_urls = 0

    for product in results:
        granule_id = product.properties.get("sceneName") or product.properties.get("fileID")
        if not granule_id:
            logging.warning("Skipping a search result without a sceneName/fileID.")
            continue
        if granule_id.lower().endswith((".h5", ".hdf5")):
            granule_id = os.path.splitext(granule_id)[0]

        # This local check is deliberately kept in addition to the server-side
        # `granule_list` filter, so only exact requested pairs are downloaded.
        track_frame_match = TRACK_FRAME_PATTERN.match(granule_id)
        if track_frame_match is None:
            logging.warning(f"Skipping granule with an unrecognised NISAR name: {granule_id}")
            continue
        track_frame_pair = (int(track_frame_match.group(1)), int(track_frame_match.group(2)))
        if track_frame_pair not in selected_track_frame_pairs:
            continue
        if polarization_mode and polarization_mode_from_filename(granule_id) != polarization_mode:
            continue  # Skip products acquired in a different polarization mode.
        if granule_id in granule_urls:  # Same granule returned twice – nothing new to add.
            continue

        urls: list[str] = []
        seen_filenames: set[str] = set()
        for url in get_product_urls(product):
            total_urls += 1
            filename = filename_from_url(url)
            if not url.lower().startswith("https://") or not filename.startswith(granule_id):
                continue  # Not a granule file (e.g. the s3credentials endpoint).
            if PRIVATE_JPL_BUCKET in url and not include_private_files:
                continue
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            urls.append(url)

        has_main_hdf5 = any(
            filename_from_url(url).lower() in (f"{granule_id}.h5".lower(), f"{granule_id}.hdf5".lower())
            for url in urls
        )
        if not has_main_hdf5:
            logging.warning(f"Skipping {granule_id}: ASF lists no main HDF5 file for it.")
            continue
        granule_urls[granule_id] = urls

    file_count = sum(len(urls) for urls in granule_urls.values())
    logging.info(
        f"Selected {len(granule_urls)} granule(s) with {file_count} file(s) out of {total_urls} "
        f"total URLs (polarization mode: {polarization_mode or 'ANY'})."
    )
    for granule_id, urls in granule_urls.items():
        logging.info(f"  {granule_id}: {len(urls)} file(s)")

    if not granule_urls:
        logging.warning("No matching NISAR granules found in search results. Exiting script.")
        sys.exit(0)

    return granule_urls

# --------------------------------------------------------------------------- #
# Download – resumable sequential downloads with retry and slow-link handling.
# --------------------------------------------------------------------------- #
def filename_from_url(url: str) -> str:
    """Return the decoded final path component without query parameters."""
    return unquote(os.path.basename(urlparse(url).path))


def content_range_total(content_range: str | None) -> int | None:
    """Extract the complete object size from an HTTP Content-Range header."""
    total_text = (content_range or "").rpartition("/")[-1]
    return int(total_text) if total_text.isdigit() else None


def response_total_bytes(response: requests.Response) -> int | None:
    """Return the full remote file size when the server provides it."""
    if response.status_code == requests.codes.partial_content:
        return content_range_total(response.headers.get("Content-Range"))
    content_length = response.headers.get("Content-Length")
    return int(content_length) if content_length and content_length.isdigit() else None


class SlowDownloadError(requests.ConnectionError):
    """Raised when a transfer remains below the configured useful speed."""


def download_single_file(url: str, output_directory: str, session: asf.ASFSession) -> None:
    """Stream one file to a resumable .part file and finalize it atomically."""
    filename = filename_from_url(url)
    destination_path = os.path.join(output_directory, filename)
    partial_path = f"{destination_path}.part"
    starting_bytes = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0
    headers = {"Range": f"bytes={starting_bytes}-"} if starting_bytes else {}

    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(Config.DOWNLOAD_CONNECT_TIMEOUT, Config.DOWNLOAD_READ_TIMEOUT),
    ) as response:
        if response.status_code == requests.codes.requested_range_not_satisfiable:
            total_bytes = content_range_total(response.headers.get("Content-Range"))
            if total_bytes is not None and starting_bytes == total_bytes:
                os.replace(partial_path, destination_path)
                return
            with open(partial_path, "wb"):
                pass
            raise IOError(
                f"Server rejected resume at byte {starting_bytes}; partial file was reset"
            )
        response.raise_for_status()

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content
        if append:
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)
            if not match or int(match.group(1)) != starting_bytes:
                with open(partial_path, "wb"):
                    pass
                raise IOError(
                    f"Unexpected Content-Range while resuming: {content_range!r}; partial file was reset"
                )
        elif starting_bytes:
            logging.warning("%s ignored its range request; restarting its partial download.", filename)
            starting_bytes = 0

        total_bytes = response_total_bytes(response)
        if total_bytes is not None and starting_bytes > total_bytes:
            with open(partial_path, "wb"):
                pass
            raise IOError(
                f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes); partial file was reset"
            )

        with open(partial_path, "ab" if append else "wb") as output_file, tqdm(
            total=total_bytes,
            initial=starting_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename[:70],
            file=sys.stdout,
            leave=True,
        ) as progress_bar:
            speed_window_started = time.monotonic()
            speed_window_bytes = 0
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                output_file.write(chunk)
                progress_bar.update(len(chunk))
                speed_window_bytes += len(chunk)

                elapsed = time.monotonic() - speed_window_started
                transfer_complete = total_bytes is not None and progress_bar.n >= total_bytes
                if (
                    Config.DOWNLOAD_MIN_SPEED > 0
                    and elapsed >= Config.DOWNLOAD_SPEED_WINDOW
                    and not transfer_complete
                ):
                    average_speed = speed_window_bytes / elapsed
                    if average_speed < Config.DOWNLOAD_MIN_SPEED:
                        raise SlowDownloadError(
                            f"Sustained speed {average_speed / 1024:.1f} KiB/s is below "
                            f"the {Config.DOWNLOAD_MIN_SPEED / 1024:.1f} KiB/s limit"
                        )
                    speed_window_started = time.monotonic()
                    speed_window_bytes = 0

    downloaded_bytes = os.path.getsize(partial_path)
    if total_bytes is not None and downloaded_bytes != total_bytes:
        raise IOError(f"Incomplete download: {downloaded_bytes} of {total_bytes} bytes received")
    os.replace(partial_path, destination_path)


def download_with_retries(
    url: str,
    output_directory: str,
    session: asf.ASFSession,
) -> str:
    """Retry an interrupted/stalled transfer while retaining downloaded bytes."""
    filename = filename_from_url(url)
    for attempt in range(1, Config.DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            download_single_file(url, output_directory, session)
            return filename
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == Config.DOWNLOAD_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{filename} failed after {attempt} attempts: {error}"
                ) from error
            wait_seconds = min(
                Config.DOWNLOAD_MAX_BACKOFF,
                Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1)),
            )
            logging.warning(
                "%s failed on attempt %d/%d: %s. Retrying from the partial file in %d seconds.",
                filename,
                attempt,
                Config.DOWNLOAD_MAX_ATTEMPTS,
                error,
                wait_seconds,
            )
            time.sleep(wait_seconds)

# --------------------------------------------------------------------------- #
# Paths – long-path support for the per-granule folders.
# --------------------------------------------------------------------------- #
def to_long_path(path: str) -> str:
    """Return a path usable beyond Windows' 260-character MAX_PATH limit.

    NISAR granule IDs are ~90 characters and appear in both the staging folder
    and the file names, so full paths easily exceed MAX_PATH on Windows.
    """
    absolute_path = os.path.abspath(path)
    if os.name == "nt" and not absolute_path.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute_path
    return absolute_path

# --------------------------------------------------------------------------- #
# Download – sequential per-granule download into one folder per granule.
# --------------------------------------------------------------------------- #
def download_granules_sequentially(
    granule_urls: dict[str, list[str]],  # {granule_id: [HDF5 + metadata + visualization URLs]}.
    output_directory: str,  # Folder that holds one <granule_id>/ sub-folder per granule.
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:
    """Download each granule's file bundle into ``<output_directory>/<granule_id>/``.

    Files are downloaded resumably, one progress bar per file.  Files already
    present are skipped, and a granule with failed files keeps whatever was
    downloaded so the next run resumes instead of starting over.
    """
    os.makedirs(output_directory, exist_ok=True)
    logging.info(f"Data download directory: {os.path.abspath(output_directory)}")
    logging.info(f"Starting sequential download of {len(granule_urls)} granule(s)...")

    downloaded_granules = 0
    skipped_granules = 0
    failed_granules = 0

    for granule_id, urls in granule_urls.items():
        granule_directory = to_long_path(os.path.join(output_directory, granule_id))
        if all(os.path.isfile(os.path.join(granule_directory, filename_from_url(url))) for url in urls):
            logging.info(f"All files already downloaded, skipping granule: {granule_id}")
            skipped_granules += 1
            continue

        os.makedirs(granule_directory, exist_ok=True)
        logging.info(f"Downloading granule {granule_id} ({len(urls)} file(s))")

        failed_files: list[str] = []
        for url in urls:
            filename = filename_from_url(url)
            destination_path = os.path.join(granule_directory, filename)

            if os.path.isfile(destination_path):
                logging.info(f"File already exists, skipping: {filename}")
                continue

            # Reuse a loose file left in the output folder by earlier versions of this script.
            legacy_path = to_long_path(os.path.join(output_directory, filename))
            if os.path.isfile(legacy_path):
                logging.info(f"Reusing previously downloaded file: {filename}")
                os.replace(legacy_path, destination_path)
                continue

            try:
                download_with_retries(url, granule_directory, session)
                logging.info(f"Successfully finished downloading file: {filename}")
            except Exception as file_error:
                failed_files.append(filename)
                logging.error(f"Failed to download {filename}: {file_error}")

        if failed_files:
            failed_granules += 1
            logging.error(
                f"Granule {granule_id} incomplete: {len(failed_files)} file(s) failed. "
                "Downloaded files are kept for the next run."
            )
            continue

        downloaded_granules += 1
        logging.info(f"Granule complete: {os.path.abspath(os.path.join(output_directory, granule_id))}")

    logging.info(
        "Download complete. Summary -> "
        f"Downloaded: {downloaded_granules}, Already complete: {skipped_granules}, Failed: {failed_granules}"
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

    try:
        normalize_polarization_mode(Config.POLARIZATION_MODE)  # Fail fast on an invalid polarization mode.
    except ValueError as mode_error:
        logging.error("Invalid polarization configuration: %s", mode_error)
        sys.exit(1)

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

    granule_urls = group_granule_urls(  # Every HDF5, metadata and visualization file per granule.
        results,
        selected_track_frame_pairs,
        Config.POLARIZATION_MODE,
        Config.INCLUDE_PRIVATE_FILES,
    )

    download_granules_sequentially(  # Download each granule's files into its own folder.
        granule_urls,
        Config.OUTPUT_DIRECTORY,
        session,
    )

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
