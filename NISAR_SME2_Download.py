"""
nisar_search_download.py

Search NASA's ASF (Alaska Satellite Facility) catalog for NISAR granules
within a given area of interest and date range, filter results down to
HDF5 product files, and download them sequentially to a local directory.

Each file download shows its own byte-level progress bar (current bytes / total bytes), 
rather than a single progress bar tracking file count.

Requirements:
    pip install asf_search tqdm requests

Credentials:
    Set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables
    before running this script.

Usage:
    python nisar_search_download.py
"""  # End of module‑level docstring – describes the whole script.
# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).    
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).  
import logging  # Standard logging module for recording execution steps, warnings, and errors.   
from datetime import datetime  # Module for handling date objects and generating dynamic timestamps. 

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
    EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME", "")  # NASA Earthdata login username.
    EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD", "")  # NASA Earthdata login password.

    LOG_DIRECTORY = "NISAR_SME2_Download_logs"  # Folder where timestamped log files are written.
    OUTPUT_DIRECTORY = "00_NISAR_L3_PR_SME2_HDF5"  # Folder where downloaded HDF5 product files are saved.

    # Area of interest in WGS84 (EPSG:4326) WKT format, passed directly to
    # the ASF search API.
    AOI_WKT = "POLYGON((96.8973 5.439,105.9604 5.439,105.9604 20.8004,96.8973 20.8004,96.8973 5.439))"

    START_DATE = datetime.strptime("2026-08-01", "%Y-%m-%d")  # Earliest acquisition date to include in the search (YYYY-MM-DD).
    END_DATE = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")  # Latest acquisition date – always today (YYYY-MM-DD).

    PRODUCT_LEVEL = "SME2"  # NISAR processing level to filter results by.

    MAX_RESULTS = 100  # Maximum number of granules the search will return.
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk, used for streaming downloads and progress updates.

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
            "Set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables before running."
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
def download_single_file(url: str, output_directory: str, session: asf.ASFSession) -> None:  # Download one file, showing a byte-level tqdm progress bar for it.
    """Download one file, showing a byte-level tqdm progress bar for it.

    Streams the response in chunks rather than loading it all into memory,
    and updates the progress bar as each chunk arrives so the bar reflects
    real download progress (not just file count).

    Args:
        url: Direct download URL for the file.
        output_directory: Local directory to save the file into.
        session: Authenticated ASFSession (subclasses requests.Session, so it can be used directly for streamed HTTP GETs).

    Raises:
        requests.HTTPError: If the server returns a non-success status code.
    """
    filename = url.split("/")[-1]  # Extract the target filename from the URL string.
    destination_path = os.path.join(output_directory, filename)  # Build the full local save path for this file.

    with session.get(url, stream=True) as response:  # Open a streamed GET request so the body isn't loaded all at once.
        response.raise_for_status()  # Raise an exception if the server returned an error status code.
        total_bytes = int(response.headers.get("Content-Length", 0))  # Read expected file size for the progress bar.

        with open(destination_path, "wb") as output_file, tqdm(  # Open the local file and a progress bar together.
            total=total_bytes,  # Progress bar's total is the file's expected size in bytes.
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
                    progress_bar.update(len(chunk))  # Advance the progress bar by the chunk's byte size.

# --------------------------------------------------------------------------- #
# Download – sequential download of many files, each with its own progress bar.               
# --------------------------------------------------------------------------- #
def download_files_sequentially(  # Download a list of files one at a time, each with its own progress bar.
    download_urls: list[str],  # URLs of the files to download.
    output_directory: str,  # Local directory to save files into (created if it doesn't already exist).
    session: asf.ASFSession,  # Authenticated ASFSession used for the HTTP requests.
) -> None:
    """Download a list of files one at a time, each with its own progress bar.

    Args:
        download_urls: URLs of the files to download.
        output_directory: Local directory to save files into (created if it doesn't already exist).
        session: Authenticated ASFSession used for the HTTP requests.
    """
    os.makedirs(output_directory, exist_ok=True)  # Create the output data directory safely if it doesn't exist.
    logging.info(f"Data download directory created at: {os.path.abspath(output_directory)}")  # Log its absolute path.
    logging.info(f"Starting sequential download of {len(download_urls)} file(s)...")  # Log start of the batch.

    successful_downloads = 0  # Initialize a counter for successful downloads.
    failed_downloads = 0  # Initialize a counter for failed downloads.

    for url in download_urls:  # Iterate through the URLs one at a time (sequential, not parallel).
        filename = url.split("/")[-1]  # Extract the filename for logging purposes.
        destination_path = os.path.join(output_directory, filename)  # Full local path where this file would be saved.

        logging.info(f"Starting download for file: {filename}")  # Log the start of this file's download.

        # Skip the file if it already exists locally – avoids re‑downloading.
        if os.path.isfile(destination_path):
            logging.info(f"File already exists, skipping: {filename}")
            successful_downloads += 1  # Count it as a successful (already‑present) download.
            continue  # Move on to the next URL.

        try:  # Begin try block for a single file download.
            download_single_file(url, output_directory, session)  # Download the file with its own progress bar.
            successful_downloads += 1  # Increment the success counter on completion.
            logging.info(f"Successfully finished downloading file: {filename}")  # Log successful completion.
        except Exception as file_error:  # Intercept any error for this specific file.
            failed_downloads += 1  # Increment the failure counter.
            logging.error(f"Failed to download {filename}: {file_error}")  # Log the detailed error message.

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

    download_files_sequentially(download_urls, Config.OUTPUT_DIRECTORY, session)  # Download each file in turn.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.                    
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
