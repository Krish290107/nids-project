"""One-command demo: starts the API and the dashboard, then streams traffic into the API.

    python scripts/run_demo.py                  # streams until you press Ctrl+C
    python scripts/run_demo.py --count 300      # stream 300 flows, then keep the servers up
    python scripts/run_demo.py --interval 0.2   # faster / slower arrival of flows

It needs the outputs of Day 3-7 (python scripts/run_preprocessing.py and
python scripts/run_training.py). Press Ctrl+C once to stop everything cleanly.
"""
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.demo import missing_prerequisites, stop_process, wait_for_http
from src.utils.paths import load_config, resolve_path


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    # treat "terminate" (closing the terminal, Task Manager, kill) like Ctrl+C, so the
    # API and dashboard are always stopped instead of being left running in the background
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    config = load_config()
    api_cfg = config["api"]
    dash_port = config["dashboard"]["port"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=config["replay"]["interval_seconds"])
    parser.add_argument("--count", type=int, default=None, help="flows to send (default: loop until Ctrl+C)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open the dashboard in a browser")
    args = parser.parse_args()

    processed = resolve_path(config["preprocessing"]["processed_dir"])
    missing = missing_prerequisites(
        [
            (resolve_path(config["training"]["model_path"]), "python scripts/run_training.py"),
            (resolve_path(config["preprocessing"]["artifact_path"]), "python scripts/run_preprocessing.py"),
            (processed / "X_test.csv", "python scripts/run_preprocessing.py"),
        ]
    )
    if missing:
        print("Cannot start the demo yet. Missing:")
        for path, hint in missing:
            print(f"  {path}\n    create it with: {hint}")
        sys.exit(1)

    api_url = f"http://{api_cfg['host']}:{api_cfg['port']}"
    dash_url = f"http://localhost:{dash_port}"
    children: list[subprocess.Popen] = []

    try:
        print("Starting the API ...")
        api = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "run_api.py")], cwd=PROJECT_ROOT)
        children.append(api)
        if not wait_for_http(f"{api_url}/health", timeout=90):
            raise SystemExit("The API did not become healthy within 90 seconds. Check the messages above.")

        print("Starting the dashboard ...")
        dash_cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "src" / "dashboard" / "app.py"), "--server.port", str(dash_port),
        ]
        if args.no_browser:
            dash_cmd += ["--server.headless", "true"]
        dash = subprocess.Popen(dash_cmd, cwd=PROJECT_ROOT)
        children.append(dash)
        if not wait_for_http(f"http://127.0.0.1:{dash_port}/_stcore/health", timeout=90):
            raise SystemExit("The dashboard did not start within 90 seconds. Check the messages above.")

        print(f"\nAPI docs:   {api_url}/docs\nDashboard:  {dash_url}\n")
        print("Streaming traffic into the API. Press Ctrl+C to stop everything.\n")

        replay_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "replay_traffic.py"), "--interval", str(args.interval)]
        replay_cmd += ["--count", str(args.count)] if args.count else ["--loop"]
        replay = subprocess.Popen(replay_cmd, cwd=PROJECT_ROOT)
        try:
            replay.wait()
        except KeyboardInterrupt:
            try:  # the replay got Ctrl+C too; give it a moment to print its summary
                replay.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_process(replay)
            return

        print(f"\nReplay finished. The servers are still running:\n  {api_url}/docs\n  {dash_url}\nPress Ctrl+C to stop them.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping servers ...")
        for child in reversed(children):
            stop_process(child)
        print("Done.")


if __name__ == "__main__":
    main()
