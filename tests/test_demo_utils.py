import http.server
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.demo import missing_prerequisites, stop_process, wait_for_http


def test_missing_prerequisites_reports_only_absent_files(tmp_path):
    present = tmp_path / "here.txt"
    present.write_text("x")
    absent = tmp_path / "gone.pkl"
    result = missing_prerequisites([(present, "make here"), (absent, "python scripts/run_training.py")])
    assert result == [(absent, "python scripts/run_training.py")]


def test_missing_prerequisites_empty_when_all_present(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert missing_prerequisites([(f, "hint")]) == []


class _OK(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # keep test output quiet
        pass


def test_wait_for_http_true_when_server_answers():
    server = http.server.HTTPServer(("127.0.0.1", 0), _OK)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert wait_for_http(f"http://127.0.0.1:{server.server_port}/", timeout=5, interval=0.1) is True
    finally:
        server.shutdown()


def test_wait_for_http_false_on_timeout_when_nothing_listens():
    start = time.monotonic()
    assert wait_for_http("http://127.0.0.1:9/health", timeout=1, interval=0.2) is False
    assert time.monotonic() - start < 5


def test_stop_process_terminates_a_running_child():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stop_process(proc, grace=5)
    assert proc.poll() is not None


def test_stop_process_is_safe_on_finished_child():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    stop_process(proc)  # must not raise
    assert proc.poll() == 0
