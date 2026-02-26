"""
Manual / on-demand tests for the DestinE retrieval tool.

These tests are NOT run by default (all marked with `destine` marker).
They require:
  - ~/.polytopeapirc token file (run desp-authentication.py first)
  - OPENAI_API_KEY environment variable (for RAG parameter search)
  - Network access to polytope API and OpenAI API
  - earthkit-data, langchain-chroma, chromadb, langchain-openai packages

Usage:
    # Run all DestinE tests
    pytest test/test_destine_tool.py -m destine -v

    # Run only search (fast, no polytope token needed — only OpenAI)
    pytest test/test_destine_tool.py -m destine -v -k search

    # Run only retrieval (needs ~/.polytopeapirc token)
    pytest test/test_destine_tool.py -m destine -v -k retrieve

    # Run with specific date range override
    DESTINE_START=20200101 DESTINE_END=20200131 pytest test/test_destine_tool.py -m destine -v -k retrieve
"""

import json
import os
import sys
import time
import uuid

import pytest

# ---------------------------------------------------------------------------
# Path setup — same pattern as era5_tool_manual.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ensure chroma_db path resolves correctly (tool uses relative paths)
os.chdir(REPO_ROOT)

from climsight.tools.destine_retrieval_tool import (
    _search_destine_parameters,
    retrieve_destine_data,
    POLYTOPEAPIRC_PATH,
)

# ---------------------------------------------------------------------------
# Markers — skip by default in normal pytest runs
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.destine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Berlin coordinates (default test location)
DEFAULT_LAT = 52.5125
DEFAULT_LON = 13.3610

# Default short date range (1 month)
DEFAULT_START = "20200101"
DEFAULT_END = "20200131"


@pytest.fixture
def polytope_token():
    """Skip if ~/.polytopeapirc token file does not exist."""
    if not POLYTOPEAPIRC_PATH.exists():
        pytest.skip(f"Polytope token not found at {POLYTOPEAPIRC_PATH}. Run desp-authentication.py first.")


@pytest.fixture
def openai_key():
    """Return OpenAI API key or skip if not set."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def thread_id():
    """Set a unique thread_id for sandbox isolation."""
    tid = f"destine-test-{uuid.uuid4().hex[:8]}"
    os.environ["CLIMSIGHT_THREAD_ID"] = tid
    yield tid
    os.environ.pop("CLIMSIGHT_THREAD_ID", None)


@pytest.fixture
def date_range():
    """Return (start, end) from env vars or defaults."""
    start = os.environ.get("DESTINE_START", DEFAULT_START)
    end = os.environ.get("DESTINE_END", DEFAULT_END)
    return start, end


# ===========================================================================
# TEST: Parameter search (RAG)
# ===========================================================================

class TestDestinESearch:
    """Tests for search_destine_parameters (RAG vector store)."""

    def test_search_temperature(self, openai_key):
        """Search for temperature — should find param 167 (2m temperature)."""
        result = _search_destine_parameters(
            query="temperature at 2 meters",
            k=5,
            openai_api_key=openai_key,
        )
        print(json.dumps(result, indent=2))

        assert result["success"] is True
        assert len(result["candidates"]) > 0

        param_ids = [c["param_id"] for c in result["candidates"]]
        print(f"Param IDs found: {param_ids}")
        # param 167 = 2m temperature — should be among top results
        assert "167" in param_ids, f"Expected param_id '167' in results, got {param_ids}"

    def test_search_precipitation(self, openai_key):
        """Search for precipitation — should find param 228 (total precipitation)."""
        result = _search_destine_parameters(
            query="total precipitation",
            k=5,
            openai_api_key=openai_key,
        )
        print(json.dumps(result, indent=2))

        assert result["success"] is True
        assert len(result["candidates"]) > 0

        param_ids = [c["param_id"] for c in result["candidates"]]
        print(f"Param IDs found: {param_ids}")

    def test_search_wind(self, openai_key):
        """Search for wind speed parameters."""
        result = _search_destine_parameters(
            query="wind speed at 10 meters",
            k=5,
            openai_api_key=openai_key,
        )
        print(json.dumps(result, indent=2))

        assert result["success"] is True
        assert len(result["candidates"]) > 0
        print(f"Wind candidates: {[(c['param_id'], c['name']) for c in result['candidates']]}")

    def test_search_sea_surface_temperature(self, openai_key):
        """Search for ocean parameter — SST."""
        result = _search_destine_parameters(
            query="sea surface temperature",
            k=5,
            openai_api_key=openai_key,
        )
        print(json.dumps(result, indent=2))

        assert result["success"] is True
        assert len(result["candidates"]) > 0

    def test_search_no_api_key(self, monkeypatch):
        """Search without API key should fail gracefully."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _search_destine_parameters(
            query="temperature",
            openai_api_key="",
        )
        assert result["success"] is False
        assert "API key" in result["error"]


# ===========================================================================
# TEST: Data retrieval (earthkit.data + polytope)
# ===========================================================================

class TestDestinERetrieval:
    """Tests for retrieve_destine_data (earthkit.data + polytope).

    These hit the live DestinE polytope endpoint — expect each call to take
    30-120 seconds depending on date range and server load.
    """

    def test_retrieve_2m_temperature_short(self, polytope_token, thread_id, date_range):
        """Download 2m temperature for a short period at Berlin."""
        start, end = date_range

        print(f"\nRetrieving param_id=167 (2m temp), levtype=sfc")
        print(f"Location: ({DEFAULT_LAT}, {DEFAULT_LON})")
        print(f"Date range: {start} to {end}")

        t0 = time.monotonic()
        result = retrieve_destine_data(
            param_id="167",
            levtype="sfc",
            start_date=start,
            end_date=end,
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(json.dumps(result, indent=2, default=str))

        assert result["success"] is True, f"Retrieval failed: {result.get('error', result.get('message'))}"
        assert "output_path_zarr" in result
        assert os.path.exists(result["output_path_zarr"]), f"Zarr not found at {result['output_path_zarr']}"

    def test_retrieve_precipitation_short(self, polytope_token, thread_id, date_range):
        """Download total precipitation for a short period at Berlin."""
        start, end = date_range

        print(f"\nRetrieving param_id=228 (total precip), levtype=sfc")

        t0 = time.monotonic()
        result = retrieve_destine_data(
            param_id="228",
            levtype="sfc",
            start_date=start,
            end_date=end,
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(json.dumps(result, indent=2, default=str))

        assert result["success"] is True, f"Retrieval failed: {result.get('error', result.get('message'))}"

    def test_retrieve_1_year(self, polytope_token, thread_id):
        """Download 2m temperature for exactly 1 year — tests larger request."""
        print(f"\nRetrieving param_id=167, 1 full year (20200101-20201231)")

        t0 = time.monotonic()
        result = retrieve_destine_data(
            param_id="167",
            levtype="sfc",
            start_date="20200101",
            end_date="20201231",
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(json.dumps(result, indent=2, default=str))

        assert result["success"] is True, f"Retrieval failed: {result.get('error', result.get('message'))}"

    def test_retrieve_2_years(self, polytope_token, thread_id):
        """Download 2m temperature for 2 years — tests the recommended max range."""
        print(f"\nRetrieving param_id=167, 2 years (20200101-20211231)")

        t0 = time.monotonic()
        result = retrieve_destine_data(
            param_id="167",
            levtype="sfc",
            start_date="20200101",
            end_date="20211231",
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(json.dumps(result, indent=2, default=str))

        assert result["success"] is True, f"Retrieval failed: {result.get('error', result.get('message'))}"

    def test_cache_hit(self, polytope_token, thread_id, date_range):
        """Second call with same params should return cached result instantly."""
        start, end = date_range

        # First call — downloads
        result1 = retrieve_destine_data(
            param_id="167", levtype="sfc",
            start_date=start, end_date=end,
            latitude=DEFAULT_LAT, longitude=DEFAULT_LON,
        )
        assert result1["success"] is True

        # Second call — should be cached
        t0 = time.monotonic()
        result2 = retrieve_destine_data(
            param_id="167", levtype="sfc",
            start_date=start, end_date=end,
            latitude=DEFAULT_LAT, longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"Cache call elapsed: {elapsed:.3f}s")
        assert result2["success"] is True
        assert elapsed < 1.0, f"Cache hit should be instant, took {elapsed:.1f}s"
        assert "Cached" in result2.get("message", "") or "Cache" in result2.get("message", "")

    def test_missing_token(self, thread_id, monkeypatch, tmp_path):
        """Should fail gracefully without token file."""
        # Temporarily point to a non-existent path
        import climsight.tools.destine_retrieval_tool as dt_module
        monkeypatch.setattr(dt_module, "POLYTOPEAPIRC_PATH", tmp_path / "nonexistent")

        result = retrieve_destine_data(
            param_id="167", levtype="sfc",
            start_date="20200101", end_date="20200131",
            latitude=DEFAULT_LAT, longitude=DEFAULT_LON,
        )
        assert result["success"] is False
        assert "token" in result["error"].lower()


# ===========================================================================
# TEST: Full workflow (search → retrieve)
# ===========================================================================

class TestDestinEWorkflow:
    """End-to-end: search for a parameter, then retrieve data."""

    def test_full_workflow_rag_to_download(self, openai_key, polytope_token, thread_id):
        """RAG lookup for '2m temperature', then download the top result."""
        # Step 1: Search
        print("\n--- Step 1: RAG Search ---")
        search_result = _search_destine_parameters(
            query="temperature at 2 meters",
            k=3,
            openai_api_key=openai_key,
        )
        assert search_result["success"] is True
        assert len(search_result["candidates"]) > 0

        top = search_result["candidates"][0]
        print(f"Top candidate: param_id={top['param_id']}, "
              f"levtype={top['levtype']}, name={top['name']}")

        # Step 2: Retrieve (1 month)
        print("\n--- Step 2: Retrieve Data ---")
        t0 = time.monotonic()
        retrieve_result = retrieve_destine_data(
            param_id=top["param_id"],
            levtype=top["levtype"],
            start_date="20200101",
            end_date="20200131",
            latitude=DEFAULT_LAT,
            longitude=DEFAULT_LON,
        )
        elapsed = time.monotonic() - t0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(json.dumps(retrieve_result, indent=2, default=str))

        assert retrieve_result["success"] is True

        # Step 3: Verify zarr can be opened
        print("\n--- Step 3: Verify Zarr ---")
        import xarray as xr
        ds = xr.open_dataset(retrieve_result["output_path_zarr"], engine="zarr")
        print(ds)
        assert len(ds.data_vars) > 0
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
