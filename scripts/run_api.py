"""Start the NIDS inference API.

Run from the project root, after Day 5-7 has produced models/model.pkl:
    python scripts/run_api.py

Then open http://127.0.0.1:8000/docs in your browser.
Stop the server with Ctrl+C.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from src.utils.paths import load_config


def main() -> None:
    api_cfg = load_config()["api"]
    # imported here so a missing model file gives its clear error at startup
    from src.api.main import app

    uvicorn.run(app, host=api_cfg["host"], port=api_cfg["port"])


if __name__ == "__main__":
    main()
