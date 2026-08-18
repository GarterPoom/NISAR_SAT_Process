# NISAR_SAT_Process

A two-stage Python pipeline for working with **NISAR** (NASA-ISRO Synthetic Aperture Radar) satellite data:

1. **`NISAR_Download.py`** — Searches NASA's ASF (Alaska Satellite Facility) catalog for NISAR granules over an area of interest (AOI) and date range, then downloads the matching HDF5 product files.
2. **`NISAR_Process.py`** — Reads the downloaded GSLC HDF5/NetCDF4 products, calculates radar intensity, applies multi-looking (speckle reduction), performs radiometric terrain correction (RTC) using a DEM, converts to decibels (dB), and exports georeferenced GeoTIFFs.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [1. Downloading Data — `NISAR_Download.py`](#1-downloading-data--nisar_downloadpy)
- [2. Processing Data — `NISAR_Process.py`](#2-processing-data--nisar_processpy)
- [Output](#output)
- [Logging](#logging)
- [Notes & Limitations](#notes--limitations)
- [License](#license)

---

## Overview

```
 AOI Shapefile ─┐
                 ├─▶ NISAR_Download.py ──▶ NISAR_Product/*.h5 ──▶ NISAR_Process.py ──▶ GeoTIFF_Processed/*.tif
 Date Range ─────┘                                                        ▲
                                                                DEM (NASA_DEM/*.tif)
```

- **Download stage**: queries the ASF `asf_search` API for NISAR granules intersecting a user-supplied shapefile AOI, filters results to HDF5 files, and downloads them with per-file byte-level progress bars.
- **Processing stage**: for each downloaded product, extracts the requested frequency/polarization layers, resamples to a 10 m grid, computes intensity, applies speckle-reduction multi-looking, applies terrain correction against a local DEM, converts to dB, and writes a tiled, compressed, overview-built GeoTIFF.

Both scripts are designed for large-area, memory-safe batch processing: downloads stream to disk in chunks, and raster processing operates tile-by-tile (512×512 windows) instead of loading full scenes into RAM.

---

## Requirements

- Python 3.9+
- A free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account (required for downloads)

### Python dependencies

| Package | Used for |
|---|---|
| `asf_search` | Querying and authenticating against the ASF catalog |
| `requests` | Streamed HTTP downloads |
| `tqdm` | Progress bars |
| `geopandas` | Reading/reprojecting the AOI shapefile |
| `h5py` | Reading NISAR HDF5/NetCDF4 products |
| `numpy` | Array math |
| `scipy` | Multi-look spatial filtering |
| `rasterio` | Reading DEMs and writing GeoTIFFs |
| `affine` | Georeferencing transforms |

## Installation

```bash
git clone <https://github.com/GarterPoom/NISAR_SAT_Process.git>
cd <NISAR_SAT_Process>
pip install asf_search requests tqdm geopandas h5py numpy scipy rasterio affine
```

> **Tip:** `geopandas` and `rasterio` have binary (GDAL) dependencies. If `pip install` fails on your platform, consider using `conda`/`mamba`:
> ```bash
> conda install -c conda-forge asf_search geopandas rasterio h5py scipy numpy affine tqdm requests
> ```

---

## Project Structure

The scripts expect (and create) the following layout, all relative to the script directory:

```
.
├── NISAR_Download.py
├── NISAR_Process.py
├── Thailand_Admin/                  # Example AOI shapefile folder (.shp + .shx + .dbf + .prj)
├── NASA_DEM/
│   └── NISAR_DEM_1-20260817_064201_Mosaic.tif   # Local DEM used for terrain correction
├── NISAR_Product/                   # Downloaded HDF5 granules land here
├── GeoTIFF_Processed/                # Processed GeoTIFF outputs land here
├── NISAR_Download_logs/              # Download run logs
└── NISAR_logs/                       # Processing run logs
```

`NISAR_Product/`, `GeoTIFF_Processed/`, and the log folders are created automatically if they don't exist.

---

## 1. Downloading Data — `NISAR_Download.py`

Searches the ASF catalog for NISAR granules over an AOI/date range and downloads them.

### What it does

1. Sets up logging (console + timestamped log file).
2. Authenticates with NASA Earthdata.
3. Loads an AOI shapefile, dissolves all features into one geometry, and reprojects it to WGS84 (EPSG:4326) if needed.
4. Queries ASF for NISAR granules matching the AOI, date range, and processing level.
5. Filters results down to direct-download HDF5 (`.h5` / `.hdf5`) URLs.
6. Downloads each file sequentially, showing a per-file byte-level progress bar, and logs a success/failure summary.

### Configuration

All settings live in the `Config` class at the top of the file:

```python
class Config:
    EARTHDATA_USERNAME = "----- your_username_of_NASA_Earthdata -----"
    EARTHDATA_PASSWORD = "----- your_password_of_NASA_Earthdata -----"

    LOG_DIRECTORY = "NISAR_Download_logs"
    OUTPUT_DIRECTORY = "NISAR_Product"

    AOI_SHAPEFILE = r"Thailand_Admin\L05_Province_ESRI_2559.shp"

    START_DATE = datetime.strptime("2026-08-01", "%Y-%m-%d")
    END_DATE = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
    PRODUCT_LEVEL = "GSLC"

    MAX_RESULTS = 100
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
```

| Setting | Description |
|---|---|
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | Your NASA Earthdata credentials. |
| `LOG_DIRECTORY` | Folder for timestamped download logs. |
| `OUTPUT_DIRECTORY` | Folder where downloaded HDF5 files are saved. |
| `AOI_SHAPEFILE` | Path to a `.shp` file defining the search area (its `.shx`/`.dbf`/`.prj` siblings must be present). |
| `START_DATE` / `END_DATE` | Acquisition date range to search. |
| `PRODUCT_LEVEL` | NISAR processing level to filter by (e.g. `GSLC`). |
| `MAX_RESULTS` | Maximum number of granules returned by the search. |
| `DOWNLOAD_CHUNK_SIZE` | Streaming download chunk size, in bytes. |

> ⚠️ **Security note:** Credentials are hardcoded for local single-machine use. Do **not** commit real credentials to GitHub — use placeholder values in the repo and fill in your own locally, or better, load them from environment variables / a `.env` file before publishing.

### Usage

```bash
python NISAR_Download.py
```

On completion, matching HDF5 granules will be in `NISAR_Product/`, and a run log will be in `NISAR_Download_logs/`.

---

## 2. Processing Data — `NISAR_Process.py`

Converts raw GSLC HDF5/NetCDF4 products into analysis-ready, georeferenced GeoTIFFs.

### Processing pipeline (per frequency/polarization layer)

1. **Read geometry** — derives an affine transform and CRS from the product's `xCoordinates`/`yCoordinates`/`projection` metadata.
2. **Windowed reprojection** — resamples the source data onto a target 10 m grid, tile by tile (512×512), to keep memory usage low regardless of scene size.
3. **Intensity calculation** — for complex SAR data, computes `Real² + Imag²` to get intensity.
4. **Multi-looking** — applies a spatial averaging filter (default: 5×5 looks) to reduce speckle noise.
5. **Radiometric Terrain Correction (RTC)** — uses a local DEM to compute terrain slope and adjusts intensity for shadow/illumination effects (`intensity / cos(incidence angle)`). Falls back to uncorrected data if the DEM is missing or a tile-level error occurs.
6. **Decibel conversion** — converts linear intensity to dB (`10 · log10(intensity)`), with non-finite values mapped to a `-9999.0` nodata value.
7. **Write GeoTIFF** — writes each tile to a compressed, internally tiled GeoTIFF, then builds pyramid overviews for fast display in GIS software.

### Configuration

Key constants near the top of the file:

```python
ROOT_DIRECTORY       = SCRIPT_DIRECTORY / "NISAR_Product"          # Input HDF5/NetCDF files
PROCESSED_DIRECTORY  = SCRIPT_DIRECTORY / "GeoTIFF_Processed"       # Output GeoTIFFs
LOG_DIRECTORY        = SCRIPT_DIRECTORY / "NISAR_logs"              # Processing logs
LOCAL_DEM_PATH        = SCRIPT_DIRECTORY / "NASA_DEM" / "NISAR_DEM_1-20260817_064201_Mosaic.tif"

GSLC_GRIDS_PATH   = "science/LSAR/GSLC/grids"   # HDF5 internal path to GSLC grids
FREQUENCIES       = ("frequencyA",)              # Frequency band(s) to process
POLARIZATIONS     = ("HH", "HV")                 # Polarization channel(s) to process
SUPPORTED_EXTENSIONS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".netcdf")

TILE_SIZE         = 512                          # Processing tile size (pixels)
OVERVIEW_FACTORS  = [2, 4, 8, 16, 32]             # GeoTIFF pyramid levels
TARGET_PIXEL_SIZE = 10.0                          # Output resolution, in meters
```

| Setting | Description |
|---|---|
| `ROOT_DIRECTORY` | Directory scanned (recursively) for supported input products. |
| `PROCESSED_DIRECTORY` | Where output GeoTIFFs are written. |
| `LOCAL_DEM_PATH` | DEM used for terrain correction. If missing, RTC is skipped and a warning is logged. |
| `GSLC_GRIDS_PATH` | Internal HDF5 group path where NISAR GSLC grids live. |
| `FREQUENCIES` / `POLARIZATIONS` | Which frequency bands and polarizations to extract and process. |
| `TILE_SIZE` | Tile size used for both reading and writing, to bound memory use. |
| `OVERVIEW_FACTORS` | Overview (pyramid) levels built into each output GeoTIFF. |
| `TARGET_PIXEL_SIZE` | Output pixel resolution in meters (default 10 m). |

### Usage

```bash
python NISAR_Process.py
```

The script will:
- Recursively scan `NISAR_Product/` for supported files.
- Process every configured frequency/polarization combination in each file.
- Skip and log any file that is incomplete/corrupted (common with interrupted downloads) or otherwise fails, then continue with the rest of the batch.

---

## Output

Each processed layer is written as:

```
GeoTIFF_Processed/<source_filename>_<frequency>_<polarization>_Processed_dB_<run_timestamp>.tif
```

Output GeoTIFF characteristics:
- Single-band, `float32`, values in **dB**
- 10 m pixel resolution (configurable)
- Nodata value: `-9999.0`
- Internally tiled (512×512 blocks), DEFLATE-compressed
- Includes built-in pyramid overviews for fast rendering in GIS software (QGIS, ArcGIS, etc.)

---

## Logging

Both scripts log to the console and to a timestamped file:

- Downloads → `NISAR_Download_logs/nisar_search_download_<timestamp>.log`
- Processing → `NISAR_logs/NISAR_L_Band_Process_<timestamp>.log`

Logs include search parameters, per-file download/processing status, and detailed error messages (with tracebacks in debug mode) for troubleshooting failed items without stopping the whole batch.

---

## Notes & Limitations

- Downloads and file processing both run **sequentially** (one file at a time), not in parallel.
- The RTC implementation is a simplified slope-based correction, not a full physically rigorous terrain-flattening model — suitable for general visualization/analysis, but review before use in precision radiometric studies.
- `NISAR_Process.py` expects the AOI/DEM to share compatible geographic extent and CRS logic; a mismatched or missing DEM causes RTC to be skipped automatically rather than the script failing.
- Ensure shapefile sidecar files (`.shx`, `.dbf`, `.prj`) are kept alongside the `.shp` referenced in `Config.AOI_SHAPEFILE`.

---

## License

Add your preferred license here (e.g., MIT, Apache 2.0).