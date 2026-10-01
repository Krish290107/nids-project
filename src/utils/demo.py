"""Small helpers for scripts/run_demo.py (kept here so they can be unit tested)."""
from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def missing_prerequisites(items: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """items: (file that must exist, command that creates it). Returns the ones that are missing."""
    return [(path, hint) for path, hint in items if not Path(path).exists()]


def wait_for_http(url: str, timeout: float = 60.0, interval: float = 0.5) -> bool:
    """Poll `url` until it answers 200, or give up after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    return False


def stop_process(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """Stop a child process politely, then forcefully. Safe to call on one that already exited."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
