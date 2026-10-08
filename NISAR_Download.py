# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).
import logging  # Standard logging module for recording execution steps, warnings, and errors.
import copy  # Copy authenticated cookies into a separate session per worker thread.
import re  # Regular expressions used to validate HTTP range responses.
import threading  # Thread-local state and a lock for concurrent progress bars.
import time  # Retry backoff delays after transient download failures.
from concurrent.futures import ThreadPoolExecutor, as_completed  # Bounded concurrent downloads.
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
    OUTPUT_DIRECTORY = "NISAR_Product"  # Folder where downloaded HDF5 product files are saved.

    # Area of interest, as a path to a shapefile (.shp). All features in the
    # file are dissolved into a single geometry and reprojected to WGS84
    # (EPSG:4326) automatically before being sent to the ASF search API.

    # Directory where this .py file lives
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Build the full path relative to that directory
    AOI_SHAPEFILE = os.path.join(script_dir,
                             "Administrative_Boundary_dir",
                             "Administrative_Boundary.shp")  # Path to the shapefile defining the area of interest (AOI).

    # Mode 1 uses the manually configured dates below. Mode 2 ignores them and
    # searches from DATE_LOOKBACK_DAYS ago through the current date and time.
    DATE_MODE = 1
    START_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    END_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Used only when DATE_MODE is 1.
    DATE_LOOKBACK_DAYS = 10

    PRODUCT_LEVEL = "GSLC"  # NISAR processing level to filter results by.
    FRAME_COVERAGE = "FULL"  # Exclude NISAR products that cover only a partial frame.
    POLARIZATION_MODE = "DH"  # "SH" (single-pol H), "DH" (dual-pol H: HH+HV), or None to keep every mode.

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    MAX_DOWNLOADS = None  # Maximum number of filtered products to download; None downloads every match.
    DOWNLOAD_CHUNK_SIZE = 256 * 1024  # 256 KB chunks keep progress and slow-link checks responsive.
    DOWNLOAD_WORKERS = 1  # Safe upper limit for simultaneous file downloads.
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
def search_nisar_granules(  # Query the ASF catalog for NISAR granules matching the given filters.
    aoi_wkt: str,  # Area of interest as a WKT geometry string.
    start_date: datetime,  # Earliest acquisition date to include.
    end_date: datetime,  # Latest acquisition date to include.
    product_level: str,  # NISAR processing level to filter on (e.g. "GSLC").
    frame_coverage: str,  # NISAR frame coverage to include ("FULL" excludes partial scenes).
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

    search_options = asf.ASFSearchOptions(  # Initialize the ASF search configuration object.
        dataset=["NISAR"],  # Filter search results to the NISAR dataset platform.
        intersectsWith=aoi_wkt,  # Filter granules intersecting the specified spatial WKT geometry.
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
# Filter – keeps HDF5 URLs, drops QA_STATS files and other polarization modes, caps the download count.
# --------------------------------------------------------------------------- #
def normalize_polarization_mode(polarization_mode: str | None) -> str | None:  # Validate the configured polarization mode.
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


def polarization_mode_from_filename(filename: str) -> str | None:  # Read the mode token from a NISAR product name.
    """Extract the frequency-A polarization mode (e.g. ``DH``) from a NISAR filename.

    NISAR names embed ``_<bandwidth>_<modeA><modeB>_`` such as ``_2005_DHDH_``;
    the first two letters describe the frequency-A polarization mode.
    """
    match = re.search(r"_\d{4}_([A-Z]{2})([A-Z]{2})_", filename)  # Locate the 4-letter mode token after the bandwidth code.
    return match.group(1) if match else None  # None when the name has no recognizable mode token.


def filter_hdf5_urls(results: asf.ASFSearchResults, polarization_mode: str | None = None) -> list[str]:  # Filter search results down to direct-download URLs for HDF5 files.
    """Filter search results down to direct-download URLs for HDF5 files.

    Args:
        results: Search results returned by search_nisar_granules().
        polarization_mode: Optional ``"SH"`` or ``"DH"``; products with another mode are dropped.

    Returns:
        A list of download URLs ending in .h5 or .hdf5, excluding any file
        whose name ends with `_QA_STATS.h5` and any product whose
        polarization mode differs from ``polarization_mode``. The list is capped at
        ``Config.MAX_DOWNLOADS`` entries when that limit is configured.

    Exits:
        Gracefully (status 0) if no HDF5 URLs are found.
    """
    polarization_mode = normalize_polarization_mode(polarization_mode)  # Validate the requested mode.
    all_urls = results.find_urls(directAccess=False)  # Extract all download URLs from the search results.

    # Build a list of URLs that meet all criteria: HDF5 extension, no QA_STATS suffix, and the requested polarization mode.
    download_urls = []
    for url in all_urls:
        filename = url.split("/")[-1]  # Extract the filename from the URL.
        if (filename.lower().endswith(('.h5', '.hdf5')) and   # Keep only HDF5 files.
            not filename.lower().endswith('_qa_stats.h5')):    # Exclude QA_STATS files.
            if polarization_mode and polarization_mode_from_filename(filename) != polarization_mode:
                continue  # Skip products acquired in a different polarization mode.
            download_urls.append(url)  # Add the URL to the list.

    logging.info(f"Extracted {len(download_urls)} HDF5 download URLs out of {len(all_urls)} total URLs (polarization mode: {polarization_mode or 'ANY'}).")  # Log counts.

    if not download_urls:  # Check if the filtered URL list is empty.
        logging.warning("No matching HDF5 (.h5 / .hdf5) files found in search results. Exiting script.")  # Log a warning.
        sys.exit(0)  # Exit script gracefully with a success status.

    if Config.MAX_DOWNLOADS is not None and len(download_urls) > Config.MAX_DOWNLOADS:
        logging.info(  # Announce the truncation so log readers understand why fewer files are downloaded than were found.
            f"Limiting download list from {len(download_urls)} to the configured maximum of {Config.MAX_DOWNLOADS}."
        )
        download_urls = download_urls[:Config.MAX_DOWNLOADS]  # Cap the number of products actually downloaded.

    return download_urls  # Return the list of filtered HDF5 download URLs.

# --------------------------------------------------------------------------- #
# Download – bounded concurrent downloads with resumable partial files.
"""Search ASF for NISAR GSLC products and download them safely in parallel.

The workflow reads an AOI from a shapefile, searches the ASF catalogue for
matching NISAR HDF5 products, and downloads those products to the configured
directory.  Downloads are streamed to ``.part`` files, resumed after an
interruption, retried with exponential backoff, and atomically renamed only
after their expected byte count has been received.

``ThreadPoolExecutor`` limits concurrent transfers to ``DOWNLOAD_WORKERS``.
Each worker owns an authenticated HTTP session, so cookie/session state is not
shared between threads.  This preserves download throughput without creating
unbounded connections, memory use, or console output contention.

Requirements:
    pip install asf_search tqdm requests geopandas shapely python-dotenv

Put the Earthdata credentials in the git-ignored ``.env`` file (copy
``.env.example``), configure the AOI shapefile, date range, output path, and
worker limit in :class:`Config`, then run ``python NISAR_Download.py``.
"""

# --------------------------------------------------------------------------- #
def filename_from_url(url: str) -> str:
    """Return a filesystem-safe product name derived from a download URL.

    The URL path is separated from any query string, reduced to its final path
    component, and URL-decoded so encoded product names are saved correctly.

    Args:
        url: Source URL for an ASF/NISAR product.

    Returns:
        Decoded filename component of ``url``.
    """
    return unquote(os.path.basename(urlparse(url).path))  # Strip URL metadata and decode escaped filename characters.


def content_range_total(content_range: str | None) -> int | None:
    """Extract the total object size from an HTTP ``Content-Range`` header.

    Args:
        content_range: Header value such as ``bytes 0-1023/2048``; may be absent.

    Returns:
        Complete object size in bytes, or ``None`` when no numeric size is supplied.
    """
    total_text = (content_range or "").rpartition("/")[-1]  # Take text after the final slash, which is the full size.
    return int(total_text) if total_text.isdigit() else None  # Convert only a valid numeric size to avoid malformed-header failures.


def response_total_bytes(response: requests.Response) -> int | None:
    """Return the remote object's full byte count when the server exposes it.

    Partial responses use ``Content-Range`` because ``Content-Length`` only
    describes the remaining segment; full responses use ``Content-Length``.

    Args:
        response: Streaming HTTP response received from the product server.

    Returns:
        Full remote object size in bytes, or ``None`` when it is unavailable.
    """
    if response.status_code == requests.codes.partial_content:
        return content_range_total(response.headers.get("Content-Range"))  # A 206 response needs its complete size from Content-Range.
    content_length = response.headers.get("Content-Length")  # Read the full-response payload size provided by the server.
    return int(content_length) if content_length and content_length.isdigit() else None  # Accept only a numeric length.


def make_worker_session_factory(authenticated_session: asf.ASFSession):
    """Build a getter that lazily creates one authenticated session per thread.

    A ``requests.Session`` is not shared by workers.  Each thread instead gets
    a private copy of the authenticated headers, cookies, and authentication
    settings, avoiding concurrent mutation of HTTP session state.

    Args:
        authenticated_session: ASF session that has already authenticated with Earthdata.

    Returns:
        Zero-argument callable returning the current worker's HTTP session.
    """
    headers = dict(authenticated_session.headers)  # Snapshot common request headers for later per-thread copies.
    cookies = copy.deepcopy(authenticated_session.cookies)  # Preserve authentication cookies without sharing a mutable cookie jar.
    thread_local = threading.local()  # Store a distinct session attribute for each download thread.

    def get_worker_session() -> requests.Session:
        """Return the calling thread's session, creating it on first use."""
        worker_session = getattr(thread_local, "session", None)
        if worker_session is None:
            worker_session = requests.Session()  # Start an isolated connection pool for this worker.
            worker_session.headers.update(headers)  # Apply the authenticated session's request headers.
            worker_session.cookies = copy.deepcopy(cookies)  # Give this worker its own copy of login cookies.
            worker_session.auth = authenticated_session.auth  # Retain any configured authentication handler.
            thread_local.session = worker_session  # Cache the initialized session in this thread only.
        return worker_session  # Reuse the thread's private session for subsequent downloads.

    return get_worker_session  # Provide the lazy getter to the thread-pool coordinator.


class SlowDownloadError(requests.ConnectionError):
    """Raised when a transfer remains below the configured useful speed."""


def download_single_file(url: str, output_directory: str, get_worker_session, progress_position: int) -> None:
    """Download one file, resume a valid partial transfer, and finalize safely.

    Data is streamed to a ``.part`` file.  If that file exists, the function
    asks the server for only the missing byte range.  A completed file is made
    visible only after its byte count is checked and ``os.replace`` atomically
    moves the partial file into its final name.

    Args:
        url: Product URL to retrieve.
        output_directory: Directory where the final product and temporary file live.
        get_worker_session: Callable returning the current worker's HTTP session.
        progress_position: Console row reserved for this file's progress bar.

    Raises:
        IOError: If server range metadata is invalid or the byte count is wrong.
        requests.RequestException: If the HTTP request cannot complete successfully.
    """
    filename = filename_from_url(url)  # Derive the local NISAR product filename from the source URL.
    destination_path = os.path.join(output_directory, filename)  # Choose the final destination path.
    partial_path = f"{destination_path}.part"  # Keep incomplete output separate from completed products.
    starting_bytes = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0  # Detect resumable data already written.
    headers = {"Range": f"bytes={starting_bytes}-"} if starting_bytes else {}  # Request only missing bytes when a partial file exists.

    with get_worker_session().get(
        url,
        headers=headers,
        stream=True,
        timeout=(Config.DOWNLOAD_CONNECT_TIMEOUT, Config.DOWNLOAD_READ_TIMEOUT),
    ) as response:
        if response.status_code == requests.codes.requested_range_not_satisfiable:
            total_bytes = content_range_total(response.headers.get("Content-Range"))  # Determine whether the local partial already has all bytes.
            if total_bytes is not None and starting_bytes == total_bytes:
                os.replace(partial_path, destination_path)  # Atomically finalize the already-complete partial file.
                return  # No additional network transfer is required.
            with open(partial_path, "wb"):
                pass  # Reset an invalid partial so the retry starts cleanly instead of repeating the same rejected range.
            raise IOError(
                f"Server rejected resume at byte {starting_bytes}; partial file was reset"
            )
        response.raise_for_status()  # Propagate non-success responses to the retry wrapper.

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content  # Append only when the server honored the range request.
        if append:
            content_range = response.headers.get("Content-Range", "")  # Read the range actually returned by the server.
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)  # Parse the returned segment's first byte.
            if not match or int(match.group(1)) != starting_bytes:
                with open(partial_path, "wb"):
                    pass  # Discard a partial that cannot be matched safely to the returned remote range.
                raise IOError(
                    f"Unexpected Content-Range while resuming: {content_range!r}; partial file was reset"
                )
        elif starting_bytes:
            logging.warning("%s ignored its range request; restarting its partial download.", filename)  # Record that the server sent the entire object instead.
            starting_bytes = 0  # Reset progress because the partial file will be overwritten.

        total_bytes = response_total_bytes(response)  # Obtain the complete expected object size when the server sends it.
        if total_bytes is not None and starting_bytes > total_bytes:
            with open(partial_path, "wb"):
                pass  # Clear a corrupt or stale partial so the next attempt can fetch the current remote object.
            raise IOError(
                f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes); partial file was reset"
            )

        with open(partial_path, "ab" if append else "wb") as output_file, tqdm(
            total=total_bytes,
            initial=starting_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename[:50],
            file=sys.stdout,
            leave=True,
            position=progress_position,
        ) as progress_bar:
            speed_window_started = time.monotonic()  # Start a rolling measurement for detecting a connection that only trickles data.
            speed_window_bytes = 0  # Count bytes received during the current slow-speed window.
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    output_file.write(chunk)  # Persist this non-empty streamed block to disk.
                    progress_bar.update(len(chunk))  # Advance the visual progress indicator by the written byte count.
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
                        speed_window_started = time.monotonic()  # Begin a fresh window after acceptable sustained throughput.
                        speed_window_bytes = 0

    downloaded_bytes = os.path.getsize(partial_path)  # Inspect the complete temporary file after the response closes.
    if total_bytes is not None and downloaded_bytes != total_bytes:
        raise IOError(f"Incomplete download: {downloaded_bytes} of {total_bytes} bytes received")  # Keep the partial file available for a later resume.
    os.replace(partial_path, destination_path)  # Atomically expose a verified file at its final path.


def download_with_retries(url: str, output_directory: str, get_worker_session, progress_position: int) -> str:
    """Retry one resumable download with exponential backoff between attempts.

    The ``.part`` file remains in place after a failed request, allowing the
    next call to ``download_single_file`` to continue from the saved byte count.

    Args:
        url: Product URL to download.
        output_directory: Directory containing final and partial downloads.
        get_worker_session: Callable returning the current worker's HTTP session.
        progress_position: Console row used by the progress bar.

    Returns:
        Filename of the successfully downloaded product.

    Raises:
        RuntimeError: If every configured transfer attempt fails.
    """
    filename = filename_from_url(url)  # Retain a readable product name for return values and log messages.
    for attempt in range(1, Config.DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            download_single_file(url, output_directory, get_worker_session, progress_position)  # Perform or resume the transfer.
            return filename  # Report success to the future monitored by the coordinator.
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == Config.DOWNLOAD_MAX_ATTEMPTS:
                raise RuntimeError(f"{filename} failed after {attempt} attempts: {error}") from error  # Surface the final failure with its original cause.
            wait_seconds = min(
                Config.DOWNLOAD_MAX_BACKOFF,
                Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1)),
            )  # Double the delay after each failure, but keep it within the configured cap.
            logging.warning(
                "%s failed on attempt %d/%d: %s. Retrying in %d seconds.",
                filename, attempt, Config.DOWNLOAD_MAX_ATTEMPTS, error, wait_seconds,
            )
            time.sleep(wait_seconds)  # Pause before retrying to reduce pressure on a transiently failing service.


def download_files_with_thread_pool(  # Coordinate bounded, concurrent product downloads.
    download_urls: list[str],  # Candidate product URLs, including any duplicates.
    output_directory: str,  # Local destination directory, created when absent.
    session: asf.ASFSession,  # Earthdata-authenticated ASF session used to seed workers.
) -> None:
    """Download product URLs concurrently with a safe worker cap.

    Existing files and duplicate URLs are skipped.  Each remaining URL runs in
    a worker with its own authenticated HTTP session.  Completion is collected
    as each future finishes, so a single failed transfer is logged without
    preventing unrelated products from completing.

    Args:
        download_urls: Candidate product URLs to process.
        output_directory: Local directory used for downloaded products.
        session: Authenticated ASF session whose credentials are copied per worker.
    """
    os.makedirs(output_directory, exist_ok=True)  # Ensure the requested local destination is ready for file output.
    logging.info(f"Data download directory created at: {os.path.abspath(output_directory)}")  # Record the fully resolved output location.
    logging.info("Starting download of %d file(s) with at most %d worker threads.", len(download_urls), Config.DOWNLOAD_WORKERS)  # Announce the workload and concurrency ceiling.

    get_worker_session = make_worker_session_factory(session)  # Create a thread-local authenticated-session provider.
    pending_urls = []  # Accumulate URLs that still need a network transfer.
    successful_downloads = 0  # Count existing and newly completed products as successes.
    for url in dict.fromkeys(download_urls):  # Preserve order while removing duplicate URLs before scheduling work.
        filename = filename_from_url(url)  # Determine the destination filename for this candidate URL.
        if os.path.isfile(os.path.join(output_directory, filename)):
            logging.info("File already exists, skipping: %s", filename)  # Avoid replacing an already completed product.
            successful_downloads += 1  # Treat a reusable existing product as a successful result.
        else:
            pending_urls.append(url)  # Queue the missing product for a worker thread.

    failed_downloads = 0  # Count files that exhaust their retry attempts.
    tqdm.set_lock(threading.RLock())  # Serialize progress-bar output from simultaneous worker threads.
    worker_count = min(Config.DOWNLOAD_WORKERS, len(pending_urls))  # Do not create more workers than pending files.
    if worker_count:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="nisar-download") as executor:
            futures = {
                executor.submit(download_with_retries, url, output_directory, get_worker_session, index % worker_count): url  # Assign each URL a retrying task and stable progress-bar row.
                for index, url in enumerate(pending_urls)  # Submit every missing URL to the bounded executor.
            }
            for future in as_completed(futures):  # Process tasks in completion order rather than submission order.
                filename = filename_from_url(futures[future])  # Recover the product name associated with this future.
                try:
                    future.result()  # Re-raise any worker exception in the coordinating thread.
                    successful_downloads += 1  # Include this newly completed file in the final summary.
                    logging.info("Successfully finished downloading file: %s", filename)  # Record individual transfer success.
                except Exception as file_error:
                    failed_downloads += 1  # Record the failure while allowing other futures to finish.
                    logging.error("Failed to download %s: %s", filename, file_error)  # Preserve the filename and root error in the log.

    logging.info(  # Emit a final operational summary after all scheduled work completes.
        f"Download complete. Summary -> Successful: {successful_downloads}, Failed: {failed_downloads}"  # Include skipped existing files as successes.
    )

# --------------------------------------------------------------------------- #
# Entry point – orchestrates the full workflow using Config settings.
# --------------------------------------------------------------------------- #
def main() -> None:  # Run the full search-and-download workflow using Config settings.
    """Run the full search-and-download workflow using Config settings."""
    setup_logging(Config.LOG_DIRECTORY)  # Set up logging before anything else runs.

    try:
        normalize_polarization_mode(Config.POLARIZATION_MODE)  # Fail fast on an invalid polarization mode.
        start_date, end_date = resolve_date_range(
            Config.DATE_MODE,
            Config.START_DATE,
            Config.END_DATE,
            Config.DATE_LOOKBACK_DAYS,
        )
    except ValueError as date_error:
        logging.error("Invalid configuration: %s", date_error)
        sys.exit(1)

    session = authenticate_earthdata(Config.EARTHDATA_USERNAME, Config.EARTHDATA_PASSWORD)  # Log in to Earthdata.

    aoi_wkt = load_aoi_wkt_from_shapefile(Config.AOI_SHAPEFILE)  # Derive the WKT search geometry from the AOI shapefile.

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,
        start_date=start_date,
        end_date=end_date,
        product_level=Config.PRODUCT_LEVEL,
        frame_coverage=Config.FRAME_COVERAGE,
        max_results=Config.MAX_RESULTS,
    )

    download_urls = filter_hdf5_urls(results, Config.POLARIZATION_MODE)  # Narrow results down to HDF5 file URLs only (excluding _QA_STATS.h5).

    download_files_with_thread_pool(download_urls, Config.OUTPUT_DIRECTORY, session)  # Download files with bounded concurrency.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
