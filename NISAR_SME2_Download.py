"""Search ASF for NISAR SME2 products and download them safely in parallel.

The script sends the configured WGS84 AOI to the ASF catalogue, filters the
resulting links to product HDF5 files, and stores the selected SME2 products
locally.  Each transfer streams directly to a ``.part`` file, resumes from a
previously received byte range when possible, retries transient failures with
exponential backoff, and atomically renames the file after completion.

The thread pool is deliberately bounded by ``Config.DOWNLOAD_WORKERS``.  Each
worker receives its own copy of the authenticated session headers and cookies,
which avoids sharing mutable HTTP session state while keeping connection and
memory pressure predictable.  Thread-safe tqdm progress bars show the active
transfers without corrupting terminal output.

Requirements:
    pip install asf_search tqdm requests

Set the Earthdata credentials, WKT AOI, date range, output directory, and
worker limit in :class:`Config`, then run ``python NISAR_SME2_Download.py``.
"""

# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).    
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).  
import logging  # Standard logging module for recording execution steps, warnings, and errors.   
import re  # Regular expressions; used to parse HTTP Content-Range headers.
import time  # Used for retry backoff delays after transient download failures.
import copy  # Copy authenticated cookies into a separate session per worker thread.
import threading  # Thread-local state and a lock for concurrent progress bars.
from concurrent.futures import ThreadPoolExecutor, as_completed  # Concurrent downloads.
from datetime import datetime  # Module for handling date objects and generating dynamic timestamps. 
from urllib.parse import unquote, urlparse  # Safely extracts filenames from download URLs.

import requests  # HTTP library; used here for its exceptions and streamed GET requests.
from tqdm import tqdm  # Library for rendering dynamic progress bars in the terminal console.
import asf_search as asf  # Alaska Satellite Facility Search Python package, imported under alias 'asf'.

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

    LOG_DIRECTORY = "NISAR_SME2_Download_logs"  # Folder where timestamped log files are written.
    OUTPUT_DIRECTORY = "your_directory_to_save_product"  # Folder where downloaded HDF5 product files are saved.

    # Area of interest in WGS84 (EPSG:4326) WKT format, passed directly to
    # the ASF search API.
    AOI_WKT = "POLYGON((96.8973 5.439,105.9604 5.439,105.9604 20.8004,96.8973 20.8004,96.8973 5.439))" # Thailand AOI WKT Example.

    START_DATE = datetime.strptime("yyyy-mm-dd", "%Y-%m-%d")  # Earliest acquisition date to include in the search (YYYY-MM-DD).
    END_DATE = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")  # Latest acquisition date – always today (YYYY-MM-DD).

    PRODUCT_LEVEL = "SME2"  # NISAR processing level to filter results by.

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk, used for streaming downloads and progress updates.
    DOWNLOAD_CONNECT_TIMEOUT = 30  # Seconds allowed to establish an HTTP connection.
    DOWNLOAD_READ_TIMEOUT = 120  # Maximum seconds with no data before treating a download as stalled.
    DOWNLOAD_MAX_ATTEMPTS = 5  # Initial attempt plus retries for interrupted/stalled downloads.
    DOWNLOAD_RETRY_BACKOFF = 5  # Base seconds between retries; increases after each failure.
    DOWNLOAD_WORKERS = 4  # Number of files downloaded concurrently.

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
# Download – downloads a single file, showing a byte‑level tqdm progress bar.               
# --------------------------------------------------------------------------- #
def filename_from_url(url: str) -> str:  # Derive a safe local name from the product URL.
    """Return the decoded URL basename, excluding query parameters."""
    return unquote(os.path.basename(urlparse(url).path))  # Parse the URL path, isolate its basename, and decode escaped characters.


def content_range_total(content_range: str | None) -> int | None:  # Extract a full object size from a range header.
    """Return the total byte count stated after ``/`` in ``Content-Range``."""
    total_text = (content_range or "").rpartition("/")[-1]  # Retain the final header component, such as ``12345``.
    return int(total_text) if total_text.isdigit() else None  # Convert a valid numeric total or report an unknown size.


def response_total_bytes(response: requests.Response) -> int | None:  # Determine the complete byte count advertised by the response.
    """Return the complete object size when the server provides it."""
    if response.status_code == requests.codes.partial_content:  # A range response reports its complete size in Content-Range.
        return content_range_total(response.headers.get("Content-Range"))  # Read that total rather than the partial payload length.
    content_length = response.headers.get("Content-Length")  # Get the full-response size if the server sends it.
    return int(content_length) if content_length and content_length.isdigit() else None  # Return a numeric size only when valid.


def make_worker_session_factory(authenticated_session: asf.ASFSession):  # Build independent authenticated sessions per thread.
    """Create an independent authenticated requests session in each worker thread."""
    headers = dict(authenticated_session.headers)  # Snapshot default/authentication headers before worker threads begin.
    cookies = copy.deepcopy(authenticated_session.cookies)  # Snapshot cookies so threads do not share a mutable cookie jar.
    thread_local = threading.local()  # Hold one private session for each executor thread.

    def get_worker_session() -> requests.Session:  # Return a cached session for the calling thread.
        worker_session = getattr(thread_local, "session", None)  # Look up only this thread's prior session.
        if worker_session is None:  # Create a session lazily the first time this thread needs one.
            worker_session = requests.Session()  # Give the thread its own connection pool and mutable state.
            worker_session.headers.update(headers)  # Apply the saved request headers.
            worker_session.cookies = copy.deepcopy(cookies)  # Copy authentication cookies into this isolated session.
            worker_session.auth = authenticated_session.auth  # Preserve any ASF authentication handler.
            thread_local.session = worker_session  # Cache the configured session for later calls in this thread.
        return worker_session  # Return the safe, authenticated thread-local session.

    return get_worker_session  # Give download tasks access to lazy per-thread session creation.


def download_single_file(url: str, output_directory: str, get_worker_session, progress_position: int) -> None:  # Stream or resume one product.
    """Download or resume one file, atomically renaming its completed .part file."""
    filename = filename_from_url(url)  # Calculate the product name once for local paths and messages.
    destination_path = os.path.join(output_directory, filename)  # Build the path exposed after a verified download.
    partial_path = f"{destination_path}.part"  # Keep incomplete data separate from usable products.
    starting_bytes = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0  # Resume at the retained byte offset.
    headers = {"Range": f"bytes={starting_bytes}-"} if starting_bytes else {}  # Request only missing bytes when resuming.

    with get_worker_session().get(  # Open a streamed request through the thread-local authenticated session.
        url,  # Download the selected product URL.
        headers=headers,  # Send a Range header only when a partial file exists.
        stream=True,  # Receive response bytes incrementally instead of holding the product in RAM.
        timeout=(Config.DOWNLOAD_CONNECT_TIMEOUT, Config.DOWNLOAD_READ_TIMEOUT),  # Bound connection setup and stalled reads.
    ) as response:  # Always close the response socket as this block exits.
        if response.status_code == requests.codes.requested_range_not_satisfiable:  # A 416 can mean the partial file already contains all bytes.
            total_bytes = content_range_total(response.headers.get("Content-Range"))  # Obtain the server's complete file size.
            if total_bytes is not None and starting_bytes == total_bytes:  # Confirm that the local partial file is exactly complete.
                os.replace(partial_path, destination_path)  # Atomically promote the verified partial file.
                return  # Finish this file without another download.
        response.raise_for_status()  # Fail this attempt for all other non-success HTTP responses.

        append = starting_bytes > 0 and response.status_code == requests.codes.partial_content  # Append only to a valid 206 range response.
        if append:  # Verify that the server started at exactly the retained byte offset.
            content_range = response.headers.get("Content-Range", "")  # Read the range the server claims to return.
            match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)  # Extract its first byte position.
            if not match or int(match.group(1)) != starting_bytes:  # Prevent corrupting a file with a mismatched range.
                raise IOError(f"Unexpected Content-Range while resuming: {content_range!r}")  # Let retry logic retain and handle it.
        elif starting_bytes:  # The server ignored a Range request and sent the entire object.
            logging.warning("%s ignored its range request; restarting its partial download.", filename)  # Explain why existing partial data is overwritten.
            starting_bytes = 0  # Reset progress to match the complete-response transfer.

        total_bytes = response_total_bytes(response)  # Read the complete expected size if the server supplies it.
        if total_bytes is not None and starting_bytes > total_bytes:  # Detect a corrupt or stale partial file before writes begin.
            raise IOError(f"Partial file is larger than server object ({starting_bytes} > {total_bytes} bytes)")  # Preserve the partial file for recovery.

        with open(partial_path, "ab" if append else "wb") as output_file, tqdm(  # Open the appropriate partial-file mode and its progress bar.
            total=total_bytes,  # Show the full byte target when known; tqdm supports unknown totals.
            initial=starting_bytes,  # Begin resumed bars at the bytes already stored.
            unit="B",  # Measure transfer progress in bytes.
            unit_scale=True,  # Present bytes as readable KiB, MiB, or GiB values.
            unit_divisor=1024,  # Use binary byte scaling.
            desc=filename[:50],  # Keep each concurrent bar's label short enough for the terminal.
            file=sys.stdout,  # Render progress to standard output.
            leave=True,  # Keep completed bar lines for an execution record.
            position=progress_position,  # Reserve a stable terminal row for this worker.
        ) as progress_bar:  # Close both resources safely at the end of the transfer.
            for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):  # Receive the response in bounded-memory chunks.
                if chunk:  # Skip empty keep-alive chunks.
                    output_file.write(chunk)  # Persist the received bytes to the partial file.
                    progress_bar.update(len(chunk))  # Advance the visual byte counter by the written size.

    downloaded_bytes = os.path.getsize(partial_path)  # Measure data safely written after the response has closed.
    if total_bytes is not None and downloaded_bytes != total_bytes:  # Reject a short or oversized transfer when its expected total is known.
        raise IOError(f"Incomplete download: {downloaded_bytes} of {total_bytes} bytes received")  # Leave .part intact for retry/resume.
    os.replace(partial_path, destination_path)  # Atomically expose the completed product only after validation.


def download_with_retries(url: str, output_directory: str, get_worker_session, progress_position: int) -> str:  # Retry one transfer without discarding progress.
    """Retry transient failures without losing already downloaded bytes."""
    filename = filename_from_url(url)  # Reuse a readable name in all retry messages.
    for attempt in range(1, Config.DOWNLOAD_MAX_ATTEMPTS + 1):  # Make one initial attempt plus the configured retries.
        try:  # Handle expected transfer failures at the per-file level.
            download_single_file(url, output_directory, get_worker_session, progress_position)  # Stream or resume this product.
            return filename  # Return success to the future collector.
        except (requests.RequestException, OSError, ValueError) as error:  # Preserve partial data after recoverable failures.
            if attempt == Config.DOWNLOAD_MAX_ATTEMPTS:  # Stop after the final permitted attempt.
                raise RuntimeError(f"{filename} failed after {attempt} attempts: {error}") from error  # Retain the original failure cause.
            wait_seconds = Config.DOWNLOAD_RETRY_BACKOFF * (2 ** (attempt - 1))  # Double the wait time after each failure.
            logging.warning(  # Record the error and planned retry delay.
                "%s failed on attempt %d/%d: %s. Retrying in %d seconds.",  # Use structured formatting for concurrent logs.
                filename, attempt, Config.DOWNLOAD_MAX_ATTEMPTS, error, wait_seconds,  # Fill the warning with file and retry details.
            )
            time.sleep(wait_seconds)  # Pause only this worker before resuming its retained partial download.

# --------------------------------------------------------------------------- #
# Download – sequential download of many files, each with its own progress bar.               
# --------------------------------------------------------------------------- #
def download_files_with_thread_pool(  # Download files concurrently with one session per worker.
    download_urls: list[str],  # URLs of the files to download.
    output_directory: str,  # Local directory to save files into (created if it doesn't already exist).
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:
    """Download files concurrently with isolated sessions, retrying interrupted transfers.

    Args:
        download_urls: URLs of the files to download.
        output_directory: Local directory to save files into (created if it doesn't already exist).
        session: Authenticated ASFSession used for the HTTP requests.
    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output data directory safely if it doesn't exist.
    logging.info(f"Data download directory created at: {os.path.abspath(output_directory)}")  # Log its absolute path.
    logging.info("Starting download of %d file(s) with %d worker threads...", len(download_urls), Config.DOWNLOAD_WORKERS)  # Record the bounded concurrency setting.

    get_worker_session = make_worker_session_factory(session)  # Create lazy, isolated authentication sessions for executor threads.
    pending_urls = []  # Collect unique files that still require a transfer.
    successful_downloads = 0  # Count completed and already-present products.
    for url in dict.fromkeys(download_urls):  # De-duplicate URLs while preserving the catalogue order.
        filename = filename_from_url(url)  # Derive the local filename used for existence checks and logging.
        if os.path.isfile(os.path.join(output_directory, filename)):  # Avoid downloading a product that is already complete.
            logging.info("File already exists, skipping: %s", filename)  # Record the intentional skip.
            successful_downloads += 1  # Treat an existing completed product as successful.
        else:  # Queue only missing products for a worker.
            pending_urls.append(url)  # Retain the URL for concurrent scheduling.

    failed_downloads = 0  # Count files that exhaust their retries without success.
    tqdm.set_lock(threading.RLock())  # Serialize terminal writes so progress bars do not overwrite one another.
    worker_count = min(Config.DOWNLOAD_WORKERS, len(pending_urls))  # Never create more threads than queued files.
    if worker_count:  # Skip executor setup when all requested products already exist.
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="nisar-download") as executor:  # Cap active network and disk transfers.
            futures = {  # Associate each submitted task with its URL for completion reporting.
                executor.submit(download_with_retries, url, output_directory, get_worker_session, index % worker_count): url  # Assign a stable progress-bar row.
                for index, url in enumerate(pending_urls)  # Submit every missing product; the executor runs only worker_count at once.
            }
            for future in as_completed(futures):  # Process each file as soon as its worker finishes.
                filename = filename_from_url(futures[future])  # Recover the product name associated with this future.
                try:  # Convert a successful or failed worker outcome into batch-level accounting.
                    future.result()  # Re-raise the worker exception here when all retries failed.
                    successful_downloads += 1  # Count a completed transfer.
                    logging.info("Successfully finished downloading file: %s", filename)  # Record the successful filename.
                except Exception as file_error:  # Keep other futures running when one product fails.
                    failed_downloads += 1  # Count the failed product.
                    logging.error("Failed to download %s: %s", filename, file_error)  # Preserve the failure reason in the log.

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

    aoi_wkt = Config.AOI_WKT  # Use the configured WKT search geometry directly.

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        product_level=Config.PRODUCT_LEVEL,
        max_results=Config.MAX_RESULTS,
    )

    download_urls = filter_hdf5_urls(results)  # Narrow results down to HDF5 file URLs only (excluding _QA_STATS.h5).

    download_files_with_thread_pool(download_urls, Config.OUTPUT_DIRECTORY, session)  # Download files concurrently with retry/resume support.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.                    
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
