"""Day 13-15 dashboard.

Start it with:
    python scripts/run_dashboard.py
(this runs `streamlit run src/dashboard/app.py` under the hood)

Reads, never writes:
    logs/predictions.db                          every prediction the API has logged
    reports/metrics/model_metrics.json           Day 5-7 evaluation results
    reports/metrics/replay_session_summary.json  Day 10-12's last replay run
    reports/figures/*.png                        confusion matrix / comparison plots
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard.data_access import overall_stats, read_json, recent_predictions_df
from src.utils.paths import load_config, resolve_path

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

config = load_config()
dash_cfg = config["dashboard"]
db_path = resolve_path(config["api"]["db_path"])
metrics_path = resolve_path(config["training"]["metrics_path"])
replay_path = resolve_path(config["replay"]["summary_path"])
figures_dir = resolve_path(config["reports"]["figures_dir"])

st.title("Network Intrusion Detection — Live Dashboard")

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh now"):
        st.rerun()
    auto_refresh = st.checkbox(f"Auto-refresh every {dash_cfg['refresh_interval_seconds']}s")
    st.caption(
        "Live data comes from logs/predictions.db, filled in by the API "
        "(`run_api.py`) and the traffic replay (`replay_traffic.py`)."
    )

tab_live, tab_model = st.tabs(["Live Monitor", "Model Performance"])

with tab_live:
    stats = overall_stats(db_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total flows logged", stats["total"])
    col2.metric("Attacks detected", stats["attacks"])
    col3.metric("Benign", stats["total"] - stats["attacks"])

    if stats["total"] == 0:
        st.info(
            "No predictions logged yet. In another terminal: "
            "`python scripts/run_api.py`, then `python scripts/replay_traffic.py`."
        )
    else:
        st.subheader("Predicted class counts")
        counts_df = pd.DataFrame(
            {"class": list(stats["per_class"].keys()), "count": list(stats["per_class"].values())}
        ).set_index("class")
        st.bar_chart(counts_df)

        st.subheader(f"Most recent flows (up to {dash_cfg['recent_predictions_limit']})")
        recent = recent_predictions_df(db_path, dash_cfg["recent_predictions_limit"])
        st.dataframe(recent, width="stretch", hide_index=True)

    replay_summary = read_json(replay_path)
    if replay_summary:
        st.subheader("Last replay session")
        st.json(replay_summary)

with tab_model:
    metrics = read_json(metrics_path)
    if metrics is None:
        st.info("No trained model metrics found yet. Run `python scripts/run_training.py` first.")
    else:
        st.subheader(f"Best model: {metrics['best_model']} (selected by {metrics['selection_metric']})")

        comparison_rows = [
            {"model": name, **{k: v for k, v in res.items() if k in ("accuracy", "macro_f1", "weighted_f1")}}
            for name, res in metrics["models"].items()
        ]
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

        best = metrics["models"][metrics["best_model"]]
        st.subheader("Per-class results")
        per_class_rows = [{"class": cls, **vals} for cls, vals in best["per_class"].items()]
        st.dataframe(pd.DataFrame(per_class_rows), width="stretch", hide_index=True)

        col_a, col_b = st.columns(2)
        cm_path = figures_dir / f"confusion_matrix_{metrics['best_model']}.png"
        if cm_path.exists():
            col_a.image(str(cm_path), caption="Confusion matrix (row-normalized)")
        comparison_png = figures_dir / "model_comparison.png"
        if comparison_png.exists():
            col_b.image(str(comparison_png), caption="Model comparison")

        fi_path = figures_dir / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), caption="Top features")

if auto_refresh:
    time.sleep(dash_cfg["refresh_interval_seconds"])
    st.rerun()
