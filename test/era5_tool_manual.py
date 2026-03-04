import json
import os
import sys
import time
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from climsight.tools.era5_retrieval_tool import retrieve_era5_data


def main() -> None:
    if not os.environ.get("CLIMSIGHT_THREAD_ID"):
        os.environ["CLIMSIGHT_THREAD_ID"] = f"era5-manual-{uuid.uuid4().hex}"

    start = time.monotonic()
    result = retrieve_era5_data(
        variable_id="2m_temperature",
        start_date="2014-01-01",
        end_date="2014-01-31",
        min_latitude=69.8265263165046,
        max_latitude=70.3265263165046,
        min_longitude=-144.19310379028323,
        max_longitude=-143.19310379028323,
    )
    elapsed = time.monotonic() - start

    print(json.dumps({"elapsed_seconds": elapsed, "result": result}, indent=2))


if __name__ == "__main__":
    main()
