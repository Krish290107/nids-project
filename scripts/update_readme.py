"""Fill the README's results section with your own latest numbers.

Run after training (and preprocessing):
    python scripts/update_readme.py

Reads reports/metrics/model_metrics.json and preprocessing_summary.json and rewrites only
the part of README.md between <!-- RESULTS:START --> and <!-- RESULTS:END -->.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_config, resolve_path
from src.utils.readme_block import render_results, replace_block


def main() -> None:
    config = load_config()
    metrics_path = resolve_path(config["training"]["metrics_path"])
    prep_path = resolve_path(config["preprocessing"]["summary_path"])
    for path, fix in [(metrics_path, "python scripts/run_training.py"), (prep_path, "python scripts/run_preprocessing.py")]:
        if not path.exists():
            raise SystemExit(f"Missing {path}\nCreate it with: {fix}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    readme_path = PROJECT_ROOT / "README.md"
    try:
        updated = replace_block(readme_path.read_text(encoding="utf-8"), render_results(metrics, prep))
    except ValueError as exc:
        raise SystemExit(str(exc))
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} with results for '{metrics['best_model']}'.")


if __name__ == "__main__":
    main()
