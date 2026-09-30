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
import streamlit.components.v1 as components

from src.dashboard.data_access import overall_stats, read_json, recent_predictions_df
from src.utils.paths import load_config, resolve_path

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

# ── Dark theme with multi-color accents ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar — magenta accent ── */
    section[data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid #D946EF !important;
    }

    /* ── Headings ── */
    h1 {
        color: #FAFAFA !important;
        font-weight: 700 !important;
        font-size: 1.65rem !important;
        letter-spacing: -0.3px;
    }
    h2, h3 {
        color: #E4E4E7 !important;
        font-weight: 600 !important;
    }

    /* ── Metric cards — magenta top border ── */
    div[data-testid="metric-container"] {
        background-color: #18181B !important;
        border: 1px solid #27272A;
        border-top: 3px solid #D946EF;
        padding: 18px 20px;
        border-radius: 8px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-top-color: #E879F9;
        transform: translateY(-2px);
    }
    div[data-testid="metric-container"] label {
        color: #A1A1AA !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    div[data-testid="stMetricValue"] {
        color: #FAFAFA !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
    }

    /* ── Tabs — blue accent ── */
    button[data-baseweb="tab"] {
        color: #71717A !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s;
    }
    button[data-baseweb="tab"]:hover {
        color: #60A5FA !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FAFAFA !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* ── Buttons — magenta accent ── */
    .stButton>button {
        border: 1px solid #D946EF;
        color: #FAFAFA;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        transition: all 0.2s ease;
        background-color: rgba(217, 70, 239, 0.08);
    }
    .stButton>button:hover {
        background-color: #D946EF !important;
        color: #000 !important;
        border-color: #D946EF !important;
    }

    /* ── Dataframes — blue left border ── */
    .stDataFrame {
        border: 1px solid #27272A !important;
        border-left: 3px solid #3B82F6 !important;
        border-radius: 6px;
    }

    /* ── Charts — cyan border ── */
    .chartjs-wrapper {
        border: 1px solid #27272A;
        border-left: 3px solid #06B6D4;
        border-radius: 6px;
        padding: 4px;
    }

    /* ── Info / Alert — blue accent ── */
    div[data-testid="stAlert"] {
        border-radius: 6px;
        border-left: 3px solid #3B82F6;
        background-color: rgba(59, 130, 246, 0.04) !important;
    }

    /* ── JSON viewer — cyan border ── */
    .stJson {
        border: 1px solid #27272A;
        border-left: 3px solid #06B6D4;
        border-radius: 6px;
    }

    /* ── Divider ── */
    hr {
        border-color: #27272A !important;
    }

    /* ── Checkbox — teal accent ── */
    .stCheckbox label span {
        color: #A1A1AA !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #09090B; }
    ::-webkit-scrollbar-thumb { background: #27272A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #D946EF; }
</style>
""", unsafe_allow_html=True)

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
    st.divider()
    st.caption(
        "Live data comes from logs/predictions.db, filled by the API "
        "(run_api.py) and traffic replay (replay_traffic.py)."
    )

tab_live, tab_model = st.tabs(["Live Monitor", "Model Performance"])

with tab_live:
    stats = overall_stats(db_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows Logged", f"{stats['total']:,}")
    col2.metric("Attacks Detected", f"{stats['attacks']:,}")
    col3.metric("Benign", f"{stats['total'] - stats['attacks']:,}")

    # Color-code each metric card with its own accent
    st.markdown("""
    <style>
        div[data-testid="column"]:nth-child(1) div[data-testid="metric-container"] {
            border-top: 3px solid #3B82F6 !important;
        }
        div[data-testid="column"]:nth-child(1) div[data-testid="stMetricValue"] {
            color: #60A5FA !important;
        }
        div[data-testid="column"]:nth-child(2) div[data-testid="metric-container"] {
            border-top: 3px solid #EF4444 !important;
        }
        div[data-testid="column"]:nth-child(2) div[data-testid="stMetricValue"] {
            color: #EF4444 !important;
        }
        div[data-testid="column"]:nth-child(3) div[data-testid="metric-container"] {
            border-top: 3px solid #22C55E !important;
        }
        div[data-testid="column"]:nth-child(3) div[data-testid="stMetricValue"] {
            color: #22C55E !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if stats["total"] == 0:
        st.info(
            "No predictions logged yet. In another terminal: "
            "`python scripts/run_api.py`, then `python scripts/replay_traffic.py`."
        )
    else:
        st.subheader("Predicted Class Counts")
        class_labels = list(stats["per_class"].keys())
        class_counts = list(stats["per_class"].values())

        n_classes_live = len(class_labels)
        cc_canvas_height = max(380, n_classes_live * 35 + 80)

        chartjs_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{ margin: 0; padding: 0; background: transparent; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: #18181B; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: #3F3F46; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #06B6D4; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
        <div style="background:#18181B; border:1px solid #27272A; border-left:3px solid #06B6D4;
                    border-radius:8px; padding:16px 20px; position: relative; height: 400px; width: 100%; overflow-y: auto; box-sizing: border-box;">
            <div style="position: relative; height: {cc_canvas_height}px; width: 100%;">
                <canvas id="classCountsChart"></canvas>
            </div>
        </div>
        <script>
            const ctx = document.getElementById('classCountsChart').getContext('2d');
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {class_labels},
                    datasets: [{{
                        label: 'Count',
                        data: {class_counts},
                        backgroundColor: function(context) {{
                            const chart = context.chart;
                            const {{ctx, chartArea}} = chart;
                            if (!chartArea) return 'rgba(6, 182, 212, 0.5)';
                            const gradient = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
                            gradient.addColorStop(0, 'rgba(6, 182, 212, 0.15)');
                            gradient.addColorStop(1, 'rgba(6, 182, 212, 0.85)');
                            return gradient;
                        }},
                        borderColor: 'rgba(6, 182, 212, 1)',
                        borderWidth: 1,
                        borderRadius: 6,
                        hoverBackgroundColor: 'rgba(6, 182, 212, 1)',
                        hoverBorderColor: '#FAFAFA',
                        hoverBorderWidth: 2
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{
                        duration: 900,
                        easing: 'easeOutQuart'
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            backgroundColor: '#27272A',
                            titleColor: '#FAFAFA',
                            bodyColor: '#A1A1AA',
                            borderColor: '#06B6D4',
                            borderWidth: 1,
                            cornerRadius: 6,
                            padding: 10,
                            titleFont: {{ family: 'Inter', weight: '600' }},
                            bodyFont: {{ family: 'Inter' }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{
                                color: '#A1A1AA',
                                font: {{ family: 'Inter', size: 11, weight: '500' }}
                            }},
                            grid: {{ color: 'rgba(39,39,42,0.5)' }}
                        }},
                        y: {{
                            ticks: {{
                                color: '#A1A1AA',
                                font: {{ family: 'Inter', size: 11 }}
                            }},
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
        </script>
        """
        components.html(chartjs_html, height=415)

        st.subheader(f"Most Recent Flows (up to {dash_cfg['recent_predictions_limit']})")
        recent = recent_predictions_df(db_path, dash_cfg["recent_predictions_limit"])
        st.dataframe(recent, use_container_width=True, hide_index=True)

    replay_summary = read_json(replay_path)
    if replay_summary:
        st.subheader("Last Replay Session")
        st.json(replay_summary)

with tab_model:
    metrics = read_json(metrics_path)
    if metrics is None:
        st.info("No trained model metrics found yet. Run `python scripts/run_training.py` first.")
    else:
        import json as _json

        st.markdown(
            f'<div style="background:#18181B; border:1px solid #27272A; border-left:3px solid #F59E0B; '
            f'border-radius:8px; padding:16px 20px; margin-bottom:16px;">'
            f'<span style="color:#A1A1AA; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.6px;">'
            f'Best Model</span><br>'
            f'<span style="color:#FAFAFA; font-size:1.3rem; font-weight:700;">'
            f'{metrics["best_model"]}</span>'
            f'<span style="color:#71717A; font-size:0.85rem; margin-left:10px;">'
            f'selected by {metrics["selection_metric"]}</span></div>',
            unsafe_allow_html=True,
        )

        comparison_rows = [
            {"model": name, **{k: v for k, v in res.items() if k in ("accuracy", "macro_f1", "weighted_f1")}}
            for name, res in metrics["models"].items()
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

        best = metrics["models"][metrics["best_model"]]
        st.subheader("Per-Class Results")
        per_class_rows = [{"class": cls, **vals} for cls, vals in best["per_class"].items()]
        st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

        # ── Chart.js: Model Comparison (grouped bar chart) ──
        st.subheader("Model Comparison")
        model_names = list(metrics["models"].keys())
        accuracy_vals = [metrics["models"][n]["accuracy"] for n in model_names]
        macro_f1_vals = [metrics["models"][n]["macro_f1"] for n in model_names]
        weighted_f1_vals = [metrics["models"][n]["weighted_f1"] for n in model_names]

        comparison_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
        <div style="background:#18181B; border:1px solid #27272A; border-left:3px solid #3B82F6;
                    border-radius:8px; padding:20px; position: relative; height: 360px; width: 100%;">
            <canvas id="modelComparisonChart"></canvas>
        </div>
        <script>
            new Chart(document.getElementById('modelComparisonChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: {_json.dumps(model_names)},
                    datasets: [
                        {{
                            label: 'Accuracy',
                            data: {accuracy_vals},
                            backgroundColor: 'rgba(59, 130, 246, 0.75)',
                            borderColor: '#3B82F6',
                            borderWidth: 1,
                            borderRadius: 4,
                            hoverBackgroundColor: '#3B82F6'
                        }},
                        {{
                            label: 'Macro F1',
                            data: {macro_f1_vals},
                            backgroundColor: 'rgba(217, 70, 239, 0.75)',
                            borderColor: '#D946EF',
                            borderWidth: 1,
                            borderRadius: 4,
                            hoverBackgroundColor: '#D946EF'
                        }},
                        {{
                            label: 'Weighted F1',
                            data: {weighted_f1_vals},
                            backgroundColor: 'rgba(6, 182, 212, 0.75)',
                            borderColor: '#06B6D4',
                            borderWidth: 1,
                            borderRadius: 4,
                            hoverBackgroundColor: '#06B6D4'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{ duration: 900, easing: 'easeOutQuart' }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Model Comparison on Held-Out Test Set',
                            color: '#FAFAFA',
                            font: {{ family: 'Inter', size: 14, weight: '600' }},
                            padding: {{ bottom: 16 }}
                        }},
                        legend: {{
                            labels: {{
                                color: '#A1A1AA',
                                font: {{ family: 'Inter', size: 11, weight: '500' }},
                                usePointStyle: true,
                                pointStyle: 'rectRounded',
                                padding: 16
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: '#27272A',
                            titleColor: '#FAFAFA',
                            bodyColor: '#A1A1AA',
                            borderColor: '#3B82F6',
                            borderWidth: 1,
                            cornerRadius: 6,
                            padding: 10,
                            titleFont: {{ family: 'Inter', weight: '600' }},
                            bodyFont: {{ family: 'Inter' }},
                            callbacks: {{
                                label: function(ctx) {{
                                    return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4);
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{ color: '#A1A1AA', font: {{ family: 'Inter', size: 11, weight: '500' }} }},
                            grid: {{ color: 'rgba(39,39,42,0.5)' }}
                        }},
                        y: {{
                            min: 0.99,
                            max: 1.001,
                            ticks: {{
                                color: '#A1A1AA',
                                font: {{ family: 'Inter', size: 11 }},
                                callback: function(v) {{ return v.toFixed(3); }}
                            }},
                            grid: {{ color: 'rgba(39,39,42,0.5)' }}
                        }}
                    }}
                }}
            }});
        </script>
        """
        components.html(comparison_html, height=400)

        # ── Chart.js: Confusion Matrix (heatmap) ──
        st.subheader("Confusion Matrix (Row-Normalized)")
        cm_raw = best["confusion_matrix"]
        class_names = list(best["per_class"].keys())
        # Row-normalize
        import numpy as np
        cm_arr = np.array(cm_raw, dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
        cm_norm_list = [[round(float(v), 4) for v in row] for row in cm_norm]

        # Build flat data array for the matrix: [{x, y, v}]
        matrix_data = []
        for i, row in enumerate(cm_norm_list):
            for j, v in enumerate(row):
                matrix_data.append({"x": j, "y": i, "v": v})

        n_classes = len(class_names)
        cm_canvas_height = max(400, 60 + 150 + n_classes * 40)
        
        cm_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{ margin: 0; padding: 0; background: transparent; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: #18181B; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: #3F3F46; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #D946EF; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
        <div style="background:#18181B; border:1px solid #27272A; border-left:3px solid #D946EF;
                    border-radius:8px; padding:20px; position: relative; width: 100%; height: 400px; overflow: auto; box-sizing: border-box;">
            <canvas id="confusionMatrixChart"></canvas>
        </div>
        <script>
        (function() {{
            const classNames = {_json.dumps(class_names)};
            const cmData = {_json.dumps(cm_norm_list)};
            const n = classNames.length;
            const canvas = document.getElementById('confusionMatrixChart');
            const ctx = canvas.getContext('2d');

            // Compute cell dimensions
            const padding = {{ top: 60, right: 30, bottom: 150, left: 160 }};

            function interpolateColor(val) {{
                // From dark (#18181B) through blue to bright cyan
                const r = Math.round(24 + (6 - 24) * val);
                const g = Math.round(24 + (182 - 24) * val);
                const b = Math.round(27 + (212 - 27) * val);
                return `rgb(${{r}},${{g}},${{b}})`;
            }}

            function draw(w, h) {{
                const cellW = (w - padding.left - padding.right) / n;
                const cellH = (h - padding.top - padding.bottom) / n;

                ctx.clearRect(0, 0, w, h);

                // Title
                ctx.fillStyle = '#FAFAFA';
                ctx.font = '600 14px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Confusion Matrix — ' + '{metrics["best_model"]}', w / 2, 28);

                // Axis labels
                ctx.fillStyle = '#A1A1AA';
                ctx.font = '500 11px Inter, sans-serif';
                ctx.fillText('Predicted', w / 2, h - 8);
                ctx.save();
                ctx.translate(14, h / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.fillText('Actual', 0, 0);
                ctx.restore();

                // Draw cells
                for (let i = 0; i < n; i++) {{
                    for (let j = 0; j < n; j++) {{
                        const val = cmData[i][j];
                        const x = padding.left + j * cellW;
                        const y = padding.top + i * cellH;

                        // Cell background
                        ctx.fillStyle = interpolateColor(val);
                        ctx.beginPath();
                        ctx.roundRect(x + 1, y + 1, cellW - 2, cellH - 2, 3);
                        ctx.fill();

                        // Cell text
                        ctx.fillStyle = val > 0.5 ? '#000000' : '#FAFAFA';
                        ctx.font = '500 11px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(val.toFixed(2), x + cellW / 2, y + cellH / 2);
                    }}
                }}

                // X-axis labels (Predicted)
                ctx.fillStyle = '#A1A1AA';
                ctx.font = '500 10px Inter, sans-serif';
                ctx.textAlign = 'right';
                for (let j = 0; j < n; j++) {{
                    const x = padding.left + j * cellW + cellW / 2;
                    const y = padding.top + n * cellH + 12;
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.rotate(-Math.PI / 4);
                    ctx.fillText(classNames[j], 0, 0);
                    ctx.restore();
                }}

                // Y-axis labels (Actual)
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                for (let i = 0; i < n; i++) {{
                    const x = padding.left - 8;
                    const y = padding.top + i * cellH + cellH / 2;
                    ctx.fillText(classNames[i], x, y);
                }}
            }}

            function resizeAndDraw() {{
                const dpr = window.devicePixelRatio || 1;
                const minW = padding.left + padding.right + n * 40;
                const minH = padding.top + padding.bottom + n * 40;
                
                let w = window.innerWidth - 42;
                if (w <= 0) {{
                    setTimeout(resizeAndDraw, 100);
                    return;
                }}
                
                const displayW = Math.max(w, minW); 
                const displayH = Math.max(400, minH);

                canvas.style.width = displayW + 'px';
                canvas.style.height = displayH + 'px';
                canvas.width = displayW * dpr;
                canvas.height = displayH * dpr;
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.scale(dpr, dpr);
                draw(displayW, displayH);
            }}

            window.addEventListener('resize', resizeAndDraw);
            resizeAndDraw();
        }})();
        </script>
        """
        components.html(cm_html, height=415)

        # ── Chart.js: Feature Importance (horizontal bar chart) ──
        top_features = metrics.get("top_features", {})
        if top_features:
            st.subheader("Top Features")
            feat_names = list(top_features.keys())[::-1]
            feat_values = list(top_features.values())[::-1]

            fi_canvas_height = max(380, len(feat_names) * 32 + 80)
            fi_html = f"""
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; background: transparent; }}
                ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
                ::-webkit-scrollbar-track {{ background: #18181B; border-radius: 4px; }}
                ::-webkit-scrollbar-thumb {{ background: #3F3F46; border-radius: 4px; }}
                ::-webkit-scrollbar-thumb:hover {{ background: #F59E0B; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
            <div style="background:#18181B; border:1px solid #27272A; border-left:3px solid #F59E0B;
                        border-radius:8px; padding:20px; position: relative; height: 400px; width: 100%; overflow-y: auto; box-sizing: border-box;">
                <div style="position: relative; height: {fi_canvas_height}px; width: 100%;">
                    <canvas id="featureImportanceChart"></canvas>
                </div>
            </div>
            <script>
            (function() {{
                const ctx = document.getElementById('featureImportanceChart').getContext('2d');
                
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {_json.dumps(feat_names)},
                        datasets: [{{
                            label: 'Importance',
                            data: {feat_values},
                            backgroundColor: function(context) {{
                                const chart = context.chart;
                                const {{ctx, chartArea}} = chart;
                                if (!chartArea) return 'rgba(245, 158, 11, 0.5)';
                                const gradient = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
                                gradient.addColorStop(0, 'rgba(245, 158, 11, 0.25)');
                                gradient.addColorStop(1, 'rgba(245, 158, 11, 0.85)');
                                return gradient;
                            }},
                            borderColor: 'rgba(245, 158, 11, 1)',
                            borderWidth: 1,
                            borderRadius: 4,
                            hoverBackgroundColor: 'rgba(245, 158, 11, 1)',
                            hoverBorderColor: '#FAFAFA',
                            hoverBorderWidth: 2
                        }}]
                    }},
                    options: {{
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {{ duration: 900, easing: 'easeOutQuart' }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Top Feature Importances — {metrics["best_model"]}',
                                color: '#FAFAFA',
                                font: {{ family: 'Inter', size: 14, weight: '600' }},
                                padding: {{ bottom: 16 }}
                            }},
                            legend: {{ display: false }},
                            tooltip: {{
                                backgroundColor: '#27272A',
                                titleColor: '#FAFAFA',
                                bodyColor: '#A1A1AA',
                                borderColor: '#F59E0B',
                                borderWidth: 1,
                                cornerRadius: 6,
                                padding: 10,
                                titleFont: {{ family: 'Inter', weight: '600' }},
                                bodyFont: {{ family: 'Inter' }},
                                callbacks: {{
                                    label: function(ctx) {{
                                        return 'Importance: ' + ctx.parsed.x.toFixed(5);
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: '#A1A1AA',
                                    font: {{ family: 'Inter', size: 11 }},
                                    callback: function(v) {{ return v.toFixed(3); }}
                                }},
                                grid: {{ color: 'rgba(39,39,42,0.5)' }}
                            }},
                            y: {{
                                ticks: {{
                                    color: '#A1A1AA',
                                    font: {{ family: 'Inter', size: 10, weight: '500' }}
                                }},
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }})();
            </script>
            """
            components.html(fi_html, height=415)

if auto_refresh:
    time.sleep(dash_cfg["refresh_interval_seconds"])
    st.rerun()
