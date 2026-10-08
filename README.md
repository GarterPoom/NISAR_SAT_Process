# NISAR_SAT_Process

Python tools for searching, downloading and processing **NISAR** (NASA-ISRO Synthetic Aperture Radar) L-band data over Thailand, and for turning the processed backscatter into flood products.

The pipeline has four parts:

1. **Plan** – the [Observation Plan](#observation-plan) tells you which Track/Frame scenes NISAR acquires over Thailand on which dates in 2026.
2. **Download** – search the ASF (Alaska Satellite Facility) catalog and download GSLC/GCOV, RSLC or SME2 products.
3. **Process** – convert HDF5 products into analysis-ready, georeferenced dB GeoTIFFs on a 5 m grid.
4. **Flood products** – build HH/HV band stacks, multi-temporal RGB composites and binary flood masks from the processed GeoTIFFs.

---

## Table of Contents

- [Workflow](#workflow)
- [Scripts](#scripts)
- [Requirements and installation](#requirements-and-installation)
- [Earthdata credentials (`.env`)](#earthdata-credentials-env)
- [Project structure](#project-structure)
- [Observation Plan](#observation-plan)
- [1. Downloading data](#1-downloading-data)
- [2. Processing data](#2-processing-data)
- [3. Flood products](#3-flood-products)
- [Output naming](#output-naming)
- [Logging](#logging)
- [Branches](#branches)
- [Notes and limitations](#notes-and-limitations)
- [License](#license)

---

## Workflow

```
 Observation Plan ──▶ pick date + Track/Frame
                              │
 AOI shapefile ───────────────┼─▶ NISAR_Download*.py ──▶ NISAR_Product/ (GSLC/GCOV)
 Date range ──────────────────┘                          NISAR_Product_RSLC/ (RSLC)
                                                                 │
              NASA_DEM/*.tif ──▶ NISAR_Process.py / NISAR_RSLC_Process.py
                                                                 │
                                                                 ▼
                                               GeoTIFF_Processed/*_Processed_dB_*.tif
                                                                 │
                    ┌────────────────────────────┬───────────────┴──────────────┐
                    ▼                            ▼                              ▼
       NISAR_Flood_Band_Stack*.py   NISAR_Flood_Multi_Temporal_*.py   NISAR_HV_Flood_From_GeoTIFF.py
         (same-date HH/HV RGB)        (before/after flood RGB)          Flood_Raster/ (0/1 masks)
```

All scripts process data in tiles/strips and stream downloads to disk, so they handle full NISAR frames without loading whole scenes into memory.

---

## Scripts

| Stage | Script | What it does |
|---|---|---|
| Download | `NISAR_Download.py` | Searches ASF for NISAR GSLC/GCOV granules intersecting the AOI shapefile, keeps full-frame HDF5 products of the chosen polarization mode, and downloads them with resumable, retrying transfers. |
| Download | `NISAR_Download_Track_Frame.py` | Same as above, but restricted to the Track/Frame pairs listed in `Config.TRACK_FRAME_PAIRS` (use the Observation Plan to choose them). |
| Download | `NISAR_Download_RSLC_Track_Frame.py` | Downloads L1 **RSLC** granules for selected Track/Frame pairs, including every companion file (QA, metadata, KML, browse PNG), into one folder per granule under `NISAR_Product_RSLC/`. |
| Download | `NISAR_SME2_Download.py` | Downloads NISAR L3 **SME2** soil-moisture products for a WKT AOI and date range with bounded parallel workers. |
| Process | `NISAR_Process.py` | Converts GSLC/GCOV HDF5/NetCDF4 into tiled dB GeoTIFFs on a 5 m grid (intensity → multilook → optional DEM-based RTC for GSLC → dB). |
| Process | `NISAR_RSLC_Process.py` | Converts radar-geometry RSLC into geocoded dB GeoTIFFs on the same 5 m grid (masking, multilook, calibration, DEM-iterated geolocation, geocoding, terrain normalization to gamma0). |
| Process + flood | `NISAR_HV_Flood_Process.py` | Processes the HV channel from HDF5 and writes both the dB GeoTIFF and a binary flood mask (`HV < -20 dB`). |
| Flood | `NISAR_HV_Flood_From_GeoTIFF.py` | Builds binary flood masks from GeoTIFFs that are already processed, without reopening the HDF5 products. |
| Flood | `NISAR_Flood_Band_Stack.py` | Same-date RGB stack: R = HV, G = HH, B = HH / HV (HH − HV in dB). |
| Flood | `NISAR_Flood_Band_Stack_Blue_HH.py` | Same-date RGB stack: R = HH, G = HV, B = HH (dB). |
| Flood | `NISAR_Flood_Multi_Temporal_RGB.py` | Two-date composite: R/B = HH low-flood date, G = HH high-flood date. |
| Flood | `NISAR_Flood_Multi_Temporal_RGB_HV_Green.py` | Two-date composite: R/B = HH low-flood date, G = HV high-flood date. |
| Flood | `NISAR_Flood_Multi_Temporal_HV_HH_RGB.py` | Two-date composite: R/B = HV low-flood date, G = HH high-flood date. |

---

## Requirements and installation

- Python 3.10+
- A free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account (needed for downloads)

| Package | Used for |
|---|---|
| `asf_search` | Searching and authenticating against the ASF catalog |
| `requests`, `tqdm` | Streamed, resumable downloads with progress bars |
| `python-dotenv` | Loading Earthdata credentials from `.env` |
| `geopandas`, `shapely` | Reading and reprojecting the AOI shapefile |
| `h5py` | Reading NISAR HDF5/NetCDF4 products |
| `numpy`, `scipy` | Array math and multilook filtering |
| `rasterio`, `affine`, `pyproj` | DEM handling, geocoding and GeoTIFF output |

```bash
git clone https://github.com/GarterPoom/NISAR_SAT_Process.git
cd NISAR_SAT_Process
conda install -c conda-forge asf_search geopandas shapely rasterio h5py numpy scipy affine pyproj tqdm requests python-dotenv
```

`geopandas` and `rasterio` depend on GDAL, so a conda-forge environment is the most reliable choice; plain `pip install` of the same packages also works where GDAL wheels are available.

---

## Earthdata credentials (`.env`)

Credentials are never stored in the code. Every download script reads them from a `.env` file next to the scripts:

```bash
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```

```ini
EARTHDATA_USERNAME='your_earthdata_username'
EARTHDATA_PASSWORD='your_earthdata_password'
```

Keep the single quotes so passwords containing `&`, `!`, `*`, `@`, `#` or `$` are read literally. `.env` is git-ignored; only `.env.example` (placeholders) is committed. Environment variables that are already set take precedence over `.env`. If either value is missing, the scripts stop with a clear error before contacting ASF.

---

## Project structure

```
.
├── NISAR_Download.py                       # GSLC/GCOV search + download (AOI)
├── NISAR_Download_Track_Frame.py           # GSLC/GCOV search + download (Track/Frame)
├── NISAR_Download_RSLC_Track_Frame.py      # RSLC + companion files (Track/Frame)
├── NISAR_SME2_Download.py                  # SME2 soil moisture download
├── NISAR_Process.py                        # GSLC/GCOV → dB GeoTIFF
├── NISAR_RSLC_Process.py                   # RSLC → geocoded dB GeoTIFF
├── NISAR_HV_Flood_Process.py               # HV → dB GeoTIFF + flood mask
├── NISAR_HV_Flood_From_GeoTIFF.py          # dB GeoTIFF → flood mask
├── NISAR_Flood_Band_Stack*.py              # Same-date HH/HV RGB stacks
├── NISAR_Flood_Multi_Temporal_*.py         # Two-date flood RGB composites
├── .env.example                            # Credential template (copy to .env)
├── Observation Plan/                       # NISAR 2026 acquisition plan for Thailand
├── Thailand_Admin/                         # AOI shapefiles (L05_Province_ESRI_2559.shp)
├── NASA_DEM/                               # DEM mosaic used for RTC / geolocation
├── NISAR_Product/                          # Downloaded GSLC/GCOV HDF5 (git-ignored)
├── NISAR_Product_RSLC/                     # Downloaded RSLC granule folders (git-ignored)
├── GeoTIFF_Processed/                      # Processed dB GeoTIFFs and RGB composites
├── Flood_Raster/                           # Binary flood masks (git-ignored)
├── NISAR_Download_logs/, NISAR_logs/       # Run logs (git-ignored)
└── soilMoisture/                           # Soil-moisture (SME2) working folder
```

Output and log folders are created automatically. Large data (HDF5 products, GeoTIFFs, DEM, shapefiles) is git-ignored; each data folder keeps a `Directory_description.txt` so the layout survives a fresh clone.

---

## Observation Plan

The `Observation Plan/` folder holds NISAR's 2026 acquisition plan cut down to Thailand. Use it to find out **when** NISAR will image an area and **which Track/Frame** to request before running a download script.

### Files

| File | Contents |
|---|---|
| `NISAR_ROP358_TFDB_ObservationPlan_CY2026-20260305.kmz` | The official NISAR Reference Observation Plan (ROP 358) Track/Frame database for calendar year 2026, released 2026-03-05. Global coverage; open in Google Earth or QGIS. |
| `NISAR_Thailand_schedule_2026.csv` | One row per planned scene over Thailand: acquisition date, pass direction, Track, Frame, scene name (`T<track>_F<frame>`), radar mode mnemonic, scene area, Thailand overlap (km² and %), centroids and bounding box. 1,729 rows. |
| `NISAR_Thailand_scene_attributes_2026.csv` | One row per Track/Frame (57 scenes) with the ROP attributes: radar mode, look angles, slant ranges, terrain statistics, EPSG, and which products are planned (`produceGSLC`, `produceGCOV`, `produceRSLC`, `produceGUNW`, `produceSMST`, ...). |
| `NISAR_Thailand_observation_plan_2026.xlsx` | The same information as a workbook: **Overview**, **Date summary** (scenes per date, ascending/descending counts, total overlap area), **Scene schedule** and **Scene attributes** sheets. |
| `NISAR_Thailand_scenes_2026.gpkg` | GeoPackage (EPSG:4326) with layers `scenes_full` (full frame footprints), `scenes_clipped` (footprints clipped to Thailand), `scene_attributes` (table) and `thailand_boundary`. |
| `orbit_paths_by_date/All_Pass/`, `Ascending/`, `Descending/` | One KMZ per acquisition date (152 dates, 2026-01-02 → 2026-12-30). Each contains three layers: full scenes intersecting Thailand, footprints clipped to Thailand, and the Thailand boundary. |

### Coverage summary (2026)

- **152 acquisition dates** from 2026-01-02 to 2026-12-30, **57 Track/Frame scenes**, 845 ascending and 884 descending scene acquisitions.
- **12-day repeat**: every Track/Frame is revisited every 12 days (30–31 acquisitions per scene in 2026), so a new pass over some part of Thailand comes every 2–3 days.
- Planned radar modes are dual-polarization (`DH`, HH + HV) 40 MHz or 20 MHz for most acquisitions, with some single-polarization (`SH`) 20 MHz passes; check `radar_mode_mnemonic` before downloading if you need HV.
- GSLC, GCOV and RSLC are planned for all 57 scenes.

| Pass | Track | Frames over Thailand | Frames with ≥ 50 % of the scene inside Thailand |
|---|---|---|---|
| Ascending | 11 | 7–12 | 9, 10, 11 |
| Ascending | 40 | 5–11 | 6 |
| Ascending | 83 | 8–12 | 9, 10 |
| Ascending | 112 | 4–12 | 9, 10, 11 |
| Ascending | 155 | 9–11 | – (edge track) |
| Descending | 4 | 79–85 | 80, 81, 82, 84 |
| Descending | 33 | 78–81 | 79 |
| Descending | 76 | 79–86 | 80, 81 |
| Descending | 105 | 78–84 | 79, 80, 81 |
| Descending | 148 | 80–86 | – (edge track) |

### Using the plan with the download scripts

1. Open `NISAR_Thailand_observation_plan_2026.xlsx` (or filter `NISAR_Thailand_schedule_2026.csv`) and find the dates and `Track`/`Frame` values covering your area. To check footprints visually, load `NISAR_Thailand_scenes_2026.gpkg` in QGIS or open the date's KMZ in `orbit_paths_by_date/`.
2. Put those pairs in `Config.TRACK_FRAME_PAIRS` of `NISAR_Download_Track_Frame.py` (GSLC/GCOV) or `NISAR_Download_RSLC_Track_Frame.py` (RSLC), e.g. `[(4, 81), (105, 80)]`.
3. Set the date window around the acquisition date (`DATE_MODE = 1` with `START_DATE`/`END_DATE`), allowing a few days for ASF to publish the product, then run the script.
4. For flood change detection, pick two acquisitions of the **same Track/Frame and pass direction**, 12 days (or a multiple of 12) apart, one before and one during the event. The multi-temporal scripts require matching Track, Frame and orbit direction.

> The plan is a forecast. Acquisitions can be re-planned, so treat the dates as expected, not guaranteed, and confirm availability with the download scripts' search step. The "Dated KMZ file" column in the workbook refers to `orbit_paths_by_date/<date>.kmz`; the files are stored in the `All_Pass/`, `Ascending/` and `Descending/` subfolders.

---

## 1. Downloading data

### `NISAR_Download.py` and `NISAR_Download_Track_Frame.py`

1. Log to the console and a timestamped file.
2. Authenticate with NASA Earthdata using the `.env` credentials.
3. Load the AOI shapefile, dissolve all features and reproject to WGS84.
4. Search ASF for NISAR granules matching AOI, dates, processing level and full-frame coverage (plus Track/Frame filename patterns in the Track/Frame script).
5. Keep only product HDF5 files (`_QA_STATS.h5` is excluded) of the configured polarization mode.
6. Download each file to a `.part` file, resume interrupted transfers with HTTP Range requests, retry stalled or slow connections with capped exponential backoff, and rename atomically when complete. Files that already exist are skipped.

Main `Config` settings:

| Setting | Description |
|---|---|
| `AOI_SHAPEFILE` | AOI shapefile (default `Thailand_Admin/L05_Province_ESRI_2559.shp`). Keep its `.shx`/`.dbf`/`.prj` alongside. |
| `DATE_MODE` | `1` = use `START_DATE`/`END_DATE`; `2` = rolling window of `DATE_LOOKBACK_DAYS` up to now. |
| `PRODUCT_LEVEL` | `GSLC`, `GCOV`, ... |
| `FRAME_COVERAGE` | `"FULL"` excludes partial-frame products. |
| `POLARIZATION_MODE` | `"DH"` (dual-pol HH+HV), `"SH"` (single-pol HH) or `None` for any (`NISAR_Download.py`). |
| `TRACK_FRAME_PAIRS` | List of `(track, frame)` tuples (`NISAR_Download_Track_Frame.py`). |
| `MAX_RESULTS` / `MAX_DOWNLOADS` | Search result limit / optional cap on files actually downloaded (`None` = no cap). |
| `DOWNLOAD_*` | Chunk size, timeouts, minimum sustained speed, retry count and backoff. |

```bash
python NISAR_Download.py
python NISAR_Download_Track_Frame.py
```

### `NISAR_Download_RSLC_Track_Frame.py`

Downloads RSLC granules for the configured Track/Frame pairs and dates. Each granule gets its own folder under `NISAR_Product_RSLC/` holding the main `.h5` and all companion files (QA stats/report/summary, ISO/JSON/YAML metadata, KML footprints, PNG browse images). Some JPL metadata files sit in a private ASF bucket and may be refused (HTTP 401/403) for ordinary accounts; with `INCLUDE_PRIVATE_JPL_FILES = True` a refusal is logged as a warning, and `False` skips them. Retries continue as long as a transfer keeps making progress (`DOWNLOAD_MAX_RETRIES` counts consecutive attempts without new bytes).

### `NISAR_SME2_Download.py`

Downloads NISAR L3 SME2 soil-moisture HDF5 for a WKT AOI (`Config.AOI_WKT`) and date range using `DOWNLOAD_WORKERS` parallel workers. Set `START_DATE`, `END_DATE` and `OUTPUT_DIRECTORY` in `Config` before running.

---

## 2. Processing data

### `NISAR_Process.py` (GSLC and GCOV)

Per frequency/polarization layer (default `frequencyA`, `HH` and `HV`):

1. **Read geometry** from `xCoordinates`/`yCoordinates`/`projection`.
2. **Resample** onto a **5 m × 5 m** grid (`TARGET_PIXEL_SIZE`), tile by tile (512 × 512).
3. **Intensity** – `|SLC|²` for GSLC; GCOV diagonal terms (`HHHH`, `HVHV`) are already power.
4. **Multilook** – spatial averaging to reduce speckle (applied to all products).
5. **RTC (GSLC only)** – slope-based terrain correction with the local DEM; GCOV is already terrain-corrected. If the DEM is missing, RTC is skipped with a warning.
6. **dB** – `10 · log10(intensity)`, non-finite values → `-9999.0` NoData.
7. **Write** a DEFLATE-compressed, tiled GeoTIFF with overviews and statistics, then publish it atomically. Per-file and whole-batch timings are logged.

```bash
python NISAR_Process.py
```

Incomplete or corrupt products (common after interrupted downloads) are logged and skipped; the batch continues.

### `NISAR_RSLC_Process.py` (RSLC)

RSLC is in radar geometry, so it is masked to valid samples, multilooked to ~20 m ground spacing, calibrated, optionally noise-corrected, geolocated with the product's geolocation grid (iterated with DEM heights), geocoded onto the 5 m map grid, terrain-normalized to **gamma0** (or sigma0/beta0) and exported in dB. Intermediate files go to `GeoTIFF_Processed/_RSLC_work/`.

```bash
python NISAR_RSLC_Process.py
```

---

## 3. Flood products

All flood scripts read the processed dB GeoTIFFs in `GeoTIFF_Processed/` and never resample: paired rasters must share CRS, transform and size.

- **Binary flood masks** – `NISAR_HV_Flood_From_GeoTIFF.py` writes `1` where the input is below the threshold (default **−20 dB**) and `0` elsewhere into `Flood_Raster/`. `NISAR_HV_Flood_Process.py` does the HV processing and masking in one run from HDF5.
- **Same-date band stacks** – `NISAR_Flood_Band_Stack.py` (B = HH − HV) and `NISAR_Flood_Band_Stack_Blue_HH.py` (B = HH) pair HH and HV files with the same source name, product, frequency and export timestamp. (`NISAR_Flood_Band_Stack.py` relaxes the timestamp rule when a source has exactly one HH and one HV file, so exports from separate runs still pair; the other script is unchanged.)
- **Multi-temporal composites** – the three `NISAR_Flood_Multi_Temporal_*` scripts pair a low-flood and a high-flood acquisition of the same Track/Frame/orbit direction. By default the earliest date is "low flood" and the latest is "high flood"; pass `--low-date YYYYMMDD --high-date YYYYMMDD` to choose known event dates. Outputs go to `GeoTIFF_Processed/Multi_Temporal_RGB/`, `Multi_Temporal_RGB_HV_Green/` and `Multi_Temporal_HV_HH_RGB/`.

```bash
python NISAR_HV_Flood_From_GeoTIFF.py
python NISAR_Flood_Band_Stack.py
python NISAR_Flood_Multi_Temporal_RGB.py --low-date 20260801 --high-date 20260906
```

In QGIS, display composites as *Multiband color* with Red = 1, Green = 2, Blue = 3. In the multi-temporal composites, areas that turn dark on the high-flood date (open water) lose green and show as magenta, while areas that brighten show as green.

---

## Output naming

```
GeoTIFF_Processed/<source_product>_<GSLC|GCOV|RSLC>_<frequency>_<polarization>_Processed_dB_<YYYYMMDD_HHMMSS>.tif
```

- Single-band `float32` in dB, 5 m pixels, NoData `-9999.0`
- 512 × 512 internal tiles, DEFLATE compression, overviews 2–32×, band statistics
- Written to a temporary `.part.tif` and renamed only when complete

---

## Logging

| Script | Log file |
|---|---|
| GSLC/GCOV download scripts | `NISAR_Download_logs/nisar_search_download_<timestamp>.log` |
| RSLC download | `NISAR_Download_logs/nisar_rslc_download_<timestamp>.log` |
| SME2 download | `NISAR_SME2_Download_logs/` |
| `NISAR_Process.py`, `NISAR_HV_Flood_Process.py` | `NISAR_logs/NISAR_L_Band_Process_<timestamp>.log` |
| `NISAR_RSLC_Process.py` | `NISAR_logs/NISAR_RSLC_Process_<timestamp>.log` |
| `NISAR_HV_Flood_From_GeoTIFF.py` | `NISAR_logs/NISAR_HV_Flood_From_GeoTIFF_<timestamp>.log` |

Logs record the search parameters, per-file status, retries, and full error details, so one failed file does not stop the batch.

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | Integrated, up-to-date code |
| `AIRBUS_PC`, `SIRPOOM_PC`, `Personal_PC` | Per-machine working branches, merged into `main` |

---

## Notes and limitations

- Downloads in the Track/Frame scripts and file processing run sequentially; `NISAR_Download.py` and `NISAR_SME2_Download.py` use a bounded worker pool (`DOWNLOAD_WORKERS`). Every downloader skips files that already exist.
- The GSLC RTC is a simplified slope-based correction, suitable for visualization and flood mapping; review it before precision radiometric work.
- The −20 dB flood threshold is a starting point; tune it per scene, polarization and land cover.
- Flood severity cannot be read from filenames. Pass `--low-date`/`--high-date` when the event dates are known.

---

## License

Add your preferred license here (e.g., MIT, Apache 2.0).
