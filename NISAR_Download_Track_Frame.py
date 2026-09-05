"""
nisar_search_download.py

Search NASA's ASF (Alaska Satellite Facility) catalog for NISAR granules
within a given area of interest and date range, filter results down to
HDF5 product files, and download them sequentially to a local directory.

Each file download shows its own byte-level progress bar (current bytes / total bytes), 
rather than a single progress bar tracking file count.

Requirements:
    pip install asf_search tqdm requests geopandas shapely

Credentials:
    Set the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables
    before running the script.

Usage:
    python nisar_search_download.py
"""  # End of module‑level docstring – describes the whole script.
# --------------------------------------------------------------------------- #
# Imports – each import gets a short comment describing its purpose.
# --------------------------------------------------------------------------- #
import os  # Module for interacting with the operating system (e.g., creating directories).    
import sys  # Module for system-specific parameters and functions (e.g., standard output, exit).  
import logging  # Standard logging module for recording execution steps, warnings, and errors.   
import re  # Regular-expression support for confirming Track/Frame values in returned filenames.
from datetime import datetime  # Module for handling date objects and generating dynamic timestamps. 

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
    EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "")  # NASA Earthdata login username.
    EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "")  # NASA Earthdata login password.

    LOG_DIRECTORY = "NISAR_Download_logs"  # Folder where timestamped log files are written.
    OUTPUT_DIRECTORY = "NISAR_Product"  # Folder where downloaded HDF5 product files are saved.

    # Area of interest, as a path to a shapefile (.shp). All features in the
    # file are dissolved into a single geometry and reprojected to WGS84
    # (EPSG:4326) automatically before being sent to the ASF search API.

    # Directory where this .py file lives
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Build the full path relative to that directory
    AOI_SHAPEFILE = os.path.join(script_dir,
                             "Thailand_Admin_Shapefile",
                             "tha_admbnda_adm1_rtsd_20190221.shp")

    START_DATE = datetime.strptime("2026-08-20", "%Y-%m-%d")  # Earliest acquisition date to include in the search (YYYY-MM-DD).
    END_DATE = datetime.strptime("2026-08-21", "%Y-%m-%d") #datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")  # Latest acquisition date – always today (YYYY-MM-DD).

    PRODUCT_LEVEL = "GSLC"  # NISAR processing level to filter results by.

    # NISAR Track/Frame pairs to download.  Add every required pair here as
    # (track, frame); for example, (105, 78) means Track 105, Frame 078.
    #
    # A pair is downloaded only when its product footprint intersects
    # AOI_SHAPEFILE.  An intersection may be either partial or complete.
    # Leave no pairs configured only if you want the script to stop before
    # searching, rather than accidentally downloading every AOI result.
    TRACK_FRAME_PAIRS: list[tuple[int, int]] = [
        (105, 79),
    ]

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
            "NASA Earthdata credentials are missing. Set EARTHDATA_USERNAME "
            "and EARTHDATA_PASSWORD before running this script."
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
    granule_patterns: list[str],  # NISAR filename patterns for selected Track/Frame pairs.
    max_results: int,  # Maximum number of granules to return.

) -> asf.ASFSearchResults:  # The raw ASF search results object.
    """Query the ASF catalog for NISAR granules matching the given filters.

    Args:
        aoi_wkt: Area of interest as a WKT geometry string.
        start_date: Earliest acquisition date to include.
        end_date: Latest acquisition date to include.
        product_level: NISAR processing level to filter on (e.g. "GSLC").
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
def filter_hdf5_urls(
    results: asf.ASFSearchResults,
    selected_track_frame_pairs: set[tuple[int, int]],
) -> list[str]:  # Filter search results down to direct-download URLs for HDF5 files.
    """Filter search results down to direct-download URLs for HDF5 files.

    Args:
        results: Search results returned by search_nisar_granules().

    Returns:
        A list of direct download URLs for selected Track/Frame pairs ending in
        .h5 or .hdf5, **excluding** any file ending with `_QA_STATS.h5`.

    Exits:
        Gracefully (status 0) if no HDF5 URLs are found.
    """
    all_urls = results.find_urls(directAccess=False)  # Extract all download URLs from the search results.

    # Capture NISAR's ..._<track>_<direction>_<frame>_... filename fields.
    # This local check is deliberately kept in addition to the server-side
    # `granule_list` filter, so only exact requested pairs are downloaded.
    track_frame_pattern = re.compile(r"^NISAR_L2_[^_]+_[^_]+_\d+_(\d+)_[AD]_(\d+)_")

    # Build a list of URLs that meet all criteria: exact Track/Frame pair,
    # HDF5 extension, and no QA statistics suffix.
    download_urls = []
    for url in all_urls:
        filename = url.split("/")[-1]  # Extract the filename from the URL.
        track_frame_match = track_frame_pattern.match(filename)
        if track_frame_match is None:
            logging.warning(f"Skipping URL with an unrecognised NISAR filename: {filename}")
            continue

        track_frame_pair = (
            int(track_frame_match.group(1)),
            int(track_frame_match.group(2)),
        )
        if (filename.lower().endswith(('.h5', '.hdf5')) and   # Keep only HDF5 files.
            not filename.lower().endswith('_qa_stats.h5') and  # Exclude QA_STATS files.
            track_frame_pair in selected_track_frame_pairs):  # Retain exact requested pairs only.
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

    aoi_wkt = load_aoi_wkt_from_shapefile(Config.AOI_SHAPEFILE)  # Derive the WKT search geometry from the AOI shapefile.
    selected_track_frame_pairs, granule_patterns = prepare_track_frame_filters(
        Config.TRACK_FRAME_PAIRS
    )

    results = search_nisar_granules(  # Run the catalog search with all configured filters.
        aoi_wkt=aoi_wkt,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        product_level=Config.PRODUCT_LEVEL,
        granule_patterns=granule_patterns,
        max_results=Config.MAX_RESULTS,
    )

    download_urls = filter_hdf5_urls(  # Narrow results to the exact pairs and HDF5 URLs (excluding _QA_STATS.h5).
        results, selected_track_frame_pairs
    )

    download_files_sequentially(download_urls, Config.OUTPUT_DIRECTORY, session)  # Download each file in turn.

    logging.info("NISAR search and download workflow completed successfully.")  # Log final completion message.

# --------------------------------------------------------------------------- #
# Script entry – ensures the script runs only when executed directly.                    
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # Check if the script is being run directly (not imported).
    main()  # Call the main function to run the workflow.
