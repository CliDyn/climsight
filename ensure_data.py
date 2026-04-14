"""
Auto-download required data files on first run (e.g. Streamlit Cloud).

Called from climsight.py before the app starts. Downloads are skipped
if the files already exist locally (normal dev setup with download_data.py).

Streamlit Cloud free tier has ~1 GB disk. We prioritise the smallest
climate source (AWI_CM ~ 30 MB) and essential geo-data, skipping the
larger nextGEMS / ICCP / DestinE archives.
"""

import os
import sys
import yaml
import glob
import zipfile
import logging
import requests

logger = logging.getLogger(__name__)

_DATA_READY = False

# ---- Priority buckets (download in this order) ----
ESSENTIAL_FILES = {
    "ne_10m_coastline.zip",
    "ne_10m_land.zip",
    "ne_10m_rivers_lake_centerlines.zip",
    "ne_10m_rivers_australia.zip",
    "ne_10m_rivers_europe.zip",
    "ne_10m_rivers_north_america.zip",
    "ne_10m_lakes.zip",
    "ne_10m_lakes_australia.zip",
    "ne_10m_lakes_europe.zip",
    "ne_10m_lakes_north_america.zip",
    "natural_hazards.zip",
    "population.zip",
}
CLIMATE_FILES = {
    "awi_cm.zip",  # ~30 MB — smallest climate dataset
}
ENRICHMENT_FILES = {
    "embedded_chunks_db_openai.zip",
    "rag_articles.zip",
    "ecocrop.zip",
}
# Intentionally skipped (too large for free Streamlit Cloud)
SKIP_FILES = {
    "IFS_9-FESOM_5-production.zip",
    "ICCP.zip",
    "DestinE.zip",
    "ipcc_text_reports.zip",
}

# ---- Per-file existence checks ----
# Maps filename -> a glob pattern that should match if that data is present.
# Used instead of checking the shared parent dir (which breaks for subdir './')
FILE_MARKERS = {
    "ne_10m_coastline.zip": "data/natural_earth/coastlines/ne_10m_coastline.shp",
    "ne_10m_land.zip": "data/natural_earth/land/ne_10m_land.shp",
    "ne_10m_rivers_lake_centerlines.zip": "data/natural_earth/rivers/ne_10m_rivers_lake_centerlines.shp",
    "ne_10m_rivers_australia.zip": "data/natural_earth/rivers/ne_10m_rivers_australia.shp",
    "ne_10m_rivers_europe.zip": "data/natural_earth/rivers/ne_10m_rivers_europe.shp",
    "ne_10m_rivers_north_america.zip": "data/natural_earth/rivers/ne_10m_rivers_north_america.shp",
    "ne_10m_lakes.zip": "data/natural_earth/lakes/ne_10m_lakes.shp",
    "ne_10m_lakes_australia.zip": "data/natural_earth/lakes/ne_10m_lakes_australia.shp",
    "ne_10m_lakes_europe.zip": "data/natural_earth/lakes/ne_10m_lakes_europe.shp",
    "ne_10m_lakes_north_america.zip": "data/natural_earth/lakes/ne_10m_lakes_north_america.shp",
    "natural_hazards.zip": "data/natural_hazards/*.csv",
    "population.zip": "data/population/*.csv",
    "awi_cm.zip": "data/*historical*.nc",
    "ecocrop.zip": "data/ecocrop/EcoCrop_DB.csv",
    "embedded_chunks_db_openai.zip": "rag_db/ipcc_reports_openai",
    "rag_articles.zip": "rag_articles",
}


def _is_already_downloaded(filename):
    """Check if a specific data file has already been downloaded."""
    marker = FILE_MARKERS.get(filename)
    if not marker:
        return False  # No marker known — always download
    # Use glob to handle wildcards
    matches = glob.glob(marker)
    if matches:
        return True
    # Also check if it's a directory
    return os.path.isdir(marker)


def _download_and_extract(url, filename, extract_to, archive_type="zip",
                          status_callback=None):
    """Download a file and extract it."""
    os.makedirs(extract_to, exist_ok=True)
    msg = f"Downloading {filename} ..."
    logger.info(msg)
    if status_callback:
        status_callback(msg)
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if status_callback and total:
                    pct = int(100 * downloaded / total)
                    status_callback(f"Downloading {filename} ... {pct}%")

        if archive_type == "zip":
            if status_callback:
                status_callback(f"Extracting {filename} ...")
            with zipfile.ZipFile(filename, "r") as zf:
                zf.extractall(extract_to)

        os.remove(filename)
        logger.info(f"  -> extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to download/extract {filename}: {e}")
        if status_callback:
            status_callback(f"Warning: failed {filename} — {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False


def ensure_data(data_sources_path="data_sources.yml", status_callback=None):
    """Download essential data if not present."""
    global _DATA_READY
    if _DATA_READY:
        return

    # Quick full-check: if AWI climate data AND land shapefile both exist, skip
    land_ok = os.path.exists("data/natural_earth/land/ne_10m_land.shp")
    climate_ok = len(glob.glob("data/*historical*.nc")) > 0
    if land_ok and climate_ok:
        logger.info("Data already present — skipping download.")
        _DATA_READY = True
        return

    logger.info("=== Data not found — starting auto-download ===")
    if status_callback:
        status_callback("Checking for required data files...")

    if not os.path.exists(data_sources_path):
        logger.error(f"Cannot find {data_sources_path}")
        return

    with open(data_sources_path, "r") as f:
        ds_config = yaml.safe_load(f)

    base_path = ds_config.get("base_path", "./data")
    sources = ds_config.get("sources", [])

    # Build ordered download list: essential -> climate -> enrichment
    ordered = []
    for bucket in [ESSENTIAL_FILES, CLIMATE_FILES, ENRICHMENT_FILES]:
        for entry in sources:
            if entry["filename"] in bucket:
                ordered.append(entry)

    total_files = len(ordered)
    for idx, entry in enumerate(ordered, 1):
        filename = entry["filename"]
        url = entry.get("url", "")
        subdir = entry.get("subdir", "./")
        archive_type = entry.get("archive_type", "zip")

        if not url:
            continue

        # Use per-file marker check instead of directory check
        if _is_already_downloaded(filename):
            logger.info(f"Already present: {filename}")
            continue

        if status_callback:
            status_callback(f"[{idx}/{total_files}] Downloading {filename} ...")

        _download_and_extract(url, filename,
                              os.path.join(base_path, subdir),
                              archive_type,
                              status_callback=status_callback)

    _DATA_READY = True
    logger.info("=== Data download complete ===")
    if status_callback:
        status_callback("Data ready.")
