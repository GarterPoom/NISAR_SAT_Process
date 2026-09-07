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
from datetime import datetime  # Module for handling date objects and generating dynamic timestamps. 
from urllib.parse import unquote, urlparse  # Safely extracts filenames from download URLs.

import requests  # HTTP library; used here for its exceptions and streamed GET requests.
from tqdm import tqdm  # Library for rendering dynamic progress bars in the terminal console.
import asf_search as asf  # Alaska Satellite Facility Search Python package, imported under alias 'asf'.
import geopandas as gpd  # Library for reading shapefiles and handling geospatial vector data.

# --------------------------------------------------------------------------- #
# Configuration – all tunable settings are gathered in this class.                
# --------------------------------------------------------------------------- #
class Config:  # Groups every tunable setting in one place instead of scattering local variables.
    """Central configuration for the search-and-download workflow.                        
    
    Keeping these values in one place makes the script easier to adapt
    (e.g., for a different AOI, date range, or product level) without
    hunting through function bodies.
    """
    EARTHDATA_USERNAME = "----- your_username_of_NASA_Earthdata -----"  # NASA Earthdata login username.                                   
    EARTHDATA_PASSWORD = "----- your_password_of_NASA_Earthdata -----"  # NASA Earthdata login password.

    LOG_DIRECTORY = "NISAR_Download_logs"  # Folder where timestamped log files are written.
    OUTPUT_DIRECTORY = "NISAR_Product"  # Folder where downloaded HDF5 product files are saved.

    # Area of interest, as a path to a shapefile (.shp). All features in the
    # file are dissolved into a single geometry and reprojected to WGS84
    # (EPSG:4326) automatically before being sent to the ASF search API.

    # Directory where this .py file lives
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Build the full path relative to that directory
    AOI_SHAPEFILE = os.path.join(script_dir,
                             "administrative_boundary_shapefile_dir",
                             "administrative.shp")

    START_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Earliest acquisition date to include in the search (YYYY-MM-DD).
    END_DATE = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")  # Latest acquisition date – always today (YYYY-MM-DD).

    PRODUCT_LEVEL = "GSLC"  # NISAR processing level to filter results by.

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk, used for streaming downloads and progress updates.
    DOWNLOAD_WORKERS = 4  # Safe upper limit for simultaneous file downloads.
    DOWNLOAD_CONNECT_TIMEOUT = 30  # Seconds allowed to establish an HTTP connection.
    DOWNLOAD_READ_TIMEOUT = 120  # Seconds allowed without receiving download data.
    DOWNLOAD_MAX_ATTEMPTS = 5  # Initial attempt plus retries for interrupted downloads.
    DOWNLOAD_RETRY_BACKOFF = 5  # Base seconds between retries; doubles after each failure.

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
    max_results: int,  # Maximum number of granules to return.

) -> asf.ASFSearchResults:  # The raw ASF search results object.
    """Query the ASF catalog for NISAR granules matching the given filters.

    Args:
        aoi_wkt: Area of interest as a WKT geometry string.
        start_date: Earliest acquisition date to include.
        end_date: Latest acquisition date to include.
        product_level: NISAR processing level to filter on (e.g. "GSLC").
        max_results: Maximum number of granules to return.

    Returns:
        The raw ASF search results object.

    Exits:
        If the search request fails.
    """
    logging.info(f"Area of Interest (AOI WKT): {aoi_wkt}")  # Log the specified spatial coverage WKT boundary.
    logging.info(f"Search date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")  # Log the search time range.
    logging.info(f"Target Processing Level: {product_level}")  # Log the selected target product level.

    search_options = asf.ASFSearchOptions(  # Initialize the ASF search configuration object.
        dataset=["NISAR"],  # Filter search results to the NISAR dataset platform.
        intersectsWith=aoi_wkt,  # Filter granules intersecting the specified spatial WKT geometry.
        start=start_date,  # Filter granules acquired on or after the start date.
        end=end_date,  # Filter granules acquired on or before the end date.
        processingLevel=[product_level],  # Filter granules by the specified product level.
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
# Filter – keeps only HDF5 URLs and excludes files ending with _QA_STATS.h5.              
# --------------------------------------------------------------------------- #
def filter_hdf5_urls(results: asf.ASFSearchResults) -> list[str]:  # Filter search results down to direct-download URLs for HDF5 files.
    """Filter search results down to direct-download URLs for HDF5 files.

    Args:
        results: Search results returned by search_nisar_granules().

    Returns:
        A list of download URLs ending in .h5 or .hdf5, **excluding** any file whose name ends with `_QA_STATS.h5`.

    Exits:
        Gracefully (status 0) if no HDF5 URLs are found.
    """
    all_urls = results.find_urls(directAccess=False)  # Extract all download URLs from the search results.

    # Build a list of URLs that meet both criteria: file extension is .h5/.hdf5 **and** filename does NOT end with _QA_STATS.h5.
    download_urls = []
    for url in all_urls:
        filename = url.split("/")[-1]  # Extract the filename from the URL.
        if (filename.lower().endswith(('.h5', '.hdf5')) and   # Keep only HDF5 files.
            not filename.lower().endswith('_qa_stats.h5')):    # Exclude QA_STATS files.
            download_urls.append(url)  # Add the URL to the list.

    logging.info(f"Extracted {len(download_urls)} HDF5 download URLs out of {len(all_urls)} total URLs.")  # Log counts.
    
    if not download_urls:  # Check if the filtered URL list is empty.
        logging.warning("No HDF5 (.h5 / .hdf5) files found in search results. Exiting script.")  # Log a warning.
        sys.exit(0)  # Exit script gracefully with a success status.

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
    pip install asf_search tqdm requests geopandas shapely

Configure the Earthdata credentials, AOI shapefile, date range, output path,
and worker limit in :class:`Config`, then run ``python NISAR_Download.py``.
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
        response.raise_for_status()  # Propagate non-success responses to the retry wrapper.

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content  # Append only when the server honored the range request.
        if append:
            content_range = response.headers.get("Content-Range", "")  # Read the range actually returned by the server.
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)  # Parse the returned segment's first byte.
            if not match or int(match.group(1)) != starting_bytes:
                raise IOError(f"Unexpected Content-Range while resuming: {content_range!r}")  # Reject data that cannot safely continue the local file.
        elif starting_bytes:
            logging.warning("%s ignored its range request; restarting its partial download.", filename)  # Record that the server sent the entire object instead.
            starting_bytes = 0  # Reset progress because the partial file will be overwritten.

        total_bytes = response_total_bytes(response)  # Obtain the complete expected object size when the server sends it.
        if total_bytes is not None and starting_bytes > total_bytes:
            raise IOError(f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes)")  # Reject a corrupt or mismatched partial file.

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
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    output_file.write(chunk)  # Persist this non-empty streamed block to disk.
                    progress_bar.update(len(chunk))  # Advance the visual progress indicator by the written byte count.

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
            wait_seconds = Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1))  # Double the delay after each failed attempt.
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

    session = authenticate_earthdata(Config.EARTHDATA_USERNAME, Config.EARTHDATA_PASSWORD)  # Log in to Earthdata.

    aoi_wkt = load_aoi_wkt_from_shapefile(Config.AOI_SHAPEFILE)  # Derive the WKT search geometry from the AOI shapefile.

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        product_level=Config.PRODUCT_LEVEL,
        max_results=Config.MAX_RESULTS,
    )

    download_urls = filter_hdf5_urls(results)  # Narrow results down to HDF5 file URLs only (excluding _QA_STATS.h5).

    download_files_with_thread_pool(download_urls, Config.OUTPUT_DIRECTORY, session)  # Download files with bounded concurrency.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.                    
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
