"""Start the Day 13-15 dashboard.

Run from the project root:
    python scripts/run_dashboard.py

This launches `streamlit run src/dashboard/app.py` for you, so you don't
need to remember the streamlit command directly. Stop with Ctrl+C.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_config


def main() -> None:
    port = load_config()["dashboard"]["port"]
    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
