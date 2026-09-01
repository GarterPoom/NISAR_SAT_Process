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
    """Extract a decoded filename without URL query parameters."""
    return unquote(os.path.basename(urlparse(url).path))


def content_range_total(content_range: str | None) -> int | None:
    """Extract the complete object size from an HTTP Content-Range header."""
    total_text = (content_range or "").rpartition("/")[-1]
    return int(total_text) if total_text.isdigit() else None


def response_total_bytes(response: requests.Response) -> int | None:
    """Return the complete object size when the server provides it."""
    if response.status_code == requests.codes.partial_content:
        return content_range_total(response.headers.get("Content-Range"))
    content_length = response.headers.get("Content-Length")
    return int(content_length) if content_length and content_length.isdigit() else None


def make_worker_session_factory(authenticated_session: asf.ASFSession):
    """Return a factory that gives each worker an independent authenticated session."""
    headers = dict(authenticated_session.headers)
    cookies = copy.deepcopy(authenticated_session.cookies)
    thread_local = threading.local()

    def get_worker_session() -> requests.Session:
        worker_session = getattr(thread_local, "session", None)
        if worker_session is None:
            worker_session = requests.Session()
            worker_session.headers.update(headers)
            worker_session.cookies = copy.deepcopy(cookies)
            worker_session.auth = authenticated_session.auth
            thread_local.session = worker_session
        return worker_session

    return get_worker_session


def download_single_file(url: str, output_directory: str, get_worker_session, progress_position: int) -> None:
    """Download or resume one file, then atomically rename its .part file."""
    filename = filename_from_url(url)
    destination_path = os.path.join(output_directory, filename)
    partial_path = f"{destination_path}.part"
    starting_bytes = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0
    headers = {"Range": f"bytes={starting_bytes}-"} if starting_bytes else {}

    with get_worker_session().get(
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
        response.raise_for_status()

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content
        if append:
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)
            if not match or int(match.group(1)) != starting_bytes:
                raise IOError(f"Unexpected Content-Range while resuming: {content_range!r}")
        elif starting_bytes:
            logging.warning("%s ignored its range request; restarting its partial download.", filename)
            starting_bytes = 0

        total_bytes = response_total_bytes(response)
        if total_bytes is not None and starting_bytes > total_bytes:
            raise IOError(f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes)")

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
                    output_file.write(chunk)
                    progress_bar.update(len(chunk))

    downloaded_bytes = os.path.getsize(partial_path)
    if total_bytes is not None and downloaded_bytes != total_bytes:
        raise IOError(f"Incomplete download: {downloaded_bytes} of {total_bytes} bytes received")
    os.replace(partial_path, destination_path)


def download_with_retries(url: str, output_directory: str, get_worker_session, progress_position: int) -> str:
    """Retry interrupted transfers without discarding their partial file."""
    filename = filename_from_url(url)
    for attempt in range(1, Config.DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            download_single_file(url, output_directory, get_worker_session, progress_position)
            return filename
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == Config.DOWNLOAD_MAX_ATTEMPTS:
                raise RuntimeError(f"{filename} failed after {attempt} attempts: {error}") from error
            wait_seconds = Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1))
            logging.warning(
                "%s failed on attempt %d/%d: %s. Retrying in %d seconds.",
                filename, attempt, Config.DOWNLOAD_MAX_ATTEMPTS, error, wait_seconds,
            )
            time.sleep(wait_seconds)


def download_files_with_thread_pool(  # Download a list of files with bounded concurrency.
    download_urls: list[str],  # URLs of the files to download.
    output_directory: str,  # Local directory to save files into (created if it doesn't already exist).
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:
    """Download files concurrently with a safe worker cap and isolated sessions.

    Args:
        download_urls: URLs of the files to download.
        output_directory: Local directory to save files into (created if it doesn't already exist).
        session: Authenticated ASFSession used for the HTTP requests.
    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output data directory safely if it doesn't exist.
    logging.info(f"Data download directory created at: {os.path.abspath(output_directory)}")  # Log its absolute path.
    logging.info("Starting download of %d file(s) with at most %d worker threads.", len(download_urls), Config.DOWNLOAD_WORKERS)

    get_worker_session = make_worker_session_factory(session)
    pending_urls = []
    successful_downloads = 0
    for url in dict.fromkeys(download_urls):  # Remove duplicate URLs before scheduling work.
        filename = filename_from_url(url)
        if os.path.isfile(os.path.join(output_directory, filename)):
            logging.info("File already exists, skipping: %s", filename)
            successful_downloads += 1
        else:
            pending_urls.append(url)

    failed_downloads = 0
    tqdm.set_lock(threading.RLock())  # Prevent simultaneous progress-bar writes from corrupting the console.
    worker_count = min(Config.DOWNLOAD_WORKERS, len(pending_urls))
    if worker_count:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="nisar-download") as executor:
            futures = {
                executor.submit(download_with_retries, url, output_directory, get_worker_session, index % worker_count): url
                for index, url in enumerate(pending_urls)
            }
            for future in as_completed(futures):
                filename = filename_from_url(futures[future])
                try:
                    future.result()
                    successful_downloads += 1
                    logging.info("Successfully finished downloading file: %s", filename)
                except Exception as file_error:
                    failed_downloads += 1
                    logging.error("Failed to download %s: %s", filename, file_error)

    logging.info(  # Log the overall download summary.
        f"Download complete. Summary -> Successful: {successful_downloads}, Failed: {failed_downloads}"
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
