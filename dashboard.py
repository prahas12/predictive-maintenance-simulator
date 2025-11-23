
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from datetime import timedelta

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.features import load_data, make_window_features

# -------------------------
# Paths and constants
# -------------------------
MODEL_PATH = Path("models") / "baseline.pkl"
DATA_PATH = Path("data") / "dataset_forced_failures.csv"
HEADER_IMAGE = "/mnt/data/afc08f84-2910-4c38-9905-85992c1f4239.png"  # optional header image in workspace

# Colors (simple, professional)
COLOR_PRIMARY = "#1F6FEB"   # blue
COLOR_ALERT = "#E02424"     # red
COLOR_SECONDARY = "#7BB6FF" # light blue
COLOR_ACCENT = "#FFB86B"    # warm accent
COLOR_TEXT = "#0B1320"
COLOR_MUTED = "#6B7280"

st.set_page_config(page_title="Equipment Health — Simple", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Minimal styling
# -------------------------
st.markdown(f"""
<style>
* {{ font-family: Inter, Roboto, Arial, sans-serif; }}
[data-testid="stAppViewContainer"] {{ background: #FFFFFF; }}
.card {{ background:#FFFFFF; padding:10px; border-radius:8px; border:1px solid rgba(0,0,0,0.06); box-shadow:0 4px 12px rgba(0,0,0,0.03); }}
.section_title {{ font-weight:600; font-size:16px; color:{COLOR_TEXT}; margin-bottom:6px; }}
.kpi {{ font-weight:700; font-size:18px; color:{COLOR_TEXT}; }}
.kpi_sub {{ font-size:12px; color:{COLOR_MUTED}; margin-top:6px; }}
.stDataFrame table {{ color: {COLOR_TEXT}; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model_safe(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data
def load_dataset_safe(path=DATA_PATH):
    try:
        return load_data(path)
    except Exception:
        return pd.DataFrame()

def smooth_trace(x, y, name, color, width=1.8):
    return go.Scatter(x=x, y=y, mode="lines", name=name,
                      line=dict(color=color, width=width))

def plot_layout(title=""):
    return dict(
        template="plotly_white",
        title=dict(text=title, font=dict(size=15, color=COLOR_TEXT)),
        xaxis=dict(color=COLOR_MUTED, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(color=COLOR_MUTED, gridcolor="rgba(0,0,0,0.05)"),
        font=dict(color=COLOR_TEXT),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

# -------------------------
# Load model + dataset
# -------------------------
pipe = load_model_safe()
df_all = load_dataset_safe()

if df_all is None or df_all.empty:
    st.error("No dataset found. Place CSV at data/dataset_forced_failures.csv or use the Upload tab.")
    st.stop()

# -------------------------
# Sidebar (user controls)
# -------------------------
st.sidebar.header("Settings")
devices = sorted(df_all["device_id"].unique())
selected_device = st.sidebar.selectbox("Device", devices)

window_sec = st.sidebar.slider("Window (s)", 30, 300, 60, step=30)
step_sec = st.sidebar.slider("Step (s)", 10, 120, 30, step=10)

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
uploaded_df = None
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        if 'timestamp' in uploaded_df.columns and not pd.api.types.is_datetime64_any_dtype(uploaded_df['timestamp']):
            uploaded_df['timestamp'] = pd.to_datetime(uploaded_df['timestamp'])
        st.sidebar.success("Uploaded file loaded. It will not change your current selection automatically.")
        if st.sidebar.button("Replace dataset with uploaded file"):
            df_all = uploaded_df
            st.sidebar.success("Dataset replaced with uploaded file. If your selected device is missing, choose another device.")
            devices = sorted(df_all["device_id"].unique())
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")

# -------------------------
# Header (title only — no caption)
# -------------------------
col1, col2 = st.columns([1, 6])
with col1:
    try:
        # use width="stretch" (Streamlit new API) for images
        st.image(HEADER_IMAGE, width="stretch")
    except Exception:
        pass
with col2:
    st.markdown(f"<h1 style='margin:0; color:{COLOR_TEXT}'>Equipment Health Dashboard</h1>", unsafe_allow_html=True)
# intentionally no caption under the title

st.markdown("")  # small spacer only

# -------------------------
# Device filter & data sanitation
# -------------------------
df_dev = df_all[df_all["device_id"] == selected_device].copy()
df_dev["timestamp"] = pd.to_datetime(df_dev["timestamp"])

# -------------------------
# KPI row (no stray empty elements)
# -------------------------
k1, k2, k3, k4 = st.columns([2, 2, 3, 3])

with k1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{selected_device}</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi_sub'>Device ID</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{len(df_dev):,}</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi_sub'>Data Points</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    span = df_dev["timestamp"].max() - df_dev["timestamp"].min()
    st.markdown(f"<div class='kpi'>{str(span)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi_sub'>Time Span</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k4:
    # compute avg risk only if model exists
    X_all, y_all, meta_all = make_window_features(df_dev, window_sec, step_sec)
    avg_risk = 0.0
    if pipe is not None and X_all.shape[0] > 0:
        try:
            probs = pipe.predict_proba(X_all)
            classes = pipe.named_steps["rf"].classes_ if hasattr(pipe, "named_steps") and "rf" in pipe.named_steps else getattr(pipe, "classes_", None)
            if classes is not None and 2 in classes:
                idx = list(classes).index(2)
                avg_risk = float(np.mean([p[idx] for p in probs]))
        except Exception:
            avg_risk = 0.0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi' style='color:{COLOR_PRIMARY}'>{avg_risk:.2%}</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi_sub'>Avg Failure Risk</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Charts
# -------------------------
st.markdown("<div class='section_title'>Sensor Streams</div>", unsafe_allow_html=True)

# Vibration chart
vib = go.Figure()
vib.add_trace(smooth_trace(df_dev["timestamp"], df_dev["vibration_x"], "Vibration X", COLOR_PRIMARY))
vib.add_trace(smooth_trace(df_dev["timestamp"], df_dev["vibration_y"], "Vibration Y", COLOR_ALERT))
vib.add_trace(smooth_trace(df_dev["timestamp"], df_dev["vibration_z"], "Vibration Z", COLOR_SECONDARY))
vib.update_layout(**plot_layout("3-Axis Vibration"))
# replace deprecated use_container_width with width="stretch"
st.plotly_chart(vib, width="stretch")

# Temperature and RPM
t1, t2 = st.columns(2)
temp = go.Figure()
temp.add_trace(smooth_trace(df_dev["timestamp"], df_dev["temperature"], "Temperature", COLOR_ACCENT))
temp.update_layout(**plot_layout("Temperature"))
t1.plotly_chart(temp, width="stretch")

rpm = go.Figure()
rpm.add_trace(smooth_trace(df_dev["timestamp"], df_dev["rpm"], "RPM", COLOR_SECONDARY))
rpm.update_layout(**plot_layout("RPM"))
t2.plotly_chart(rpm, width="stretch")

# -------------------------
# Risk windows (table)
# -------------------------
st.markdown("<div class='section_title'>Risk Windows</div>", unsafe_allow_html=True)

if pipe is not None and X_all.shape[0] > 0:
    try:
        classes = pipe.named_steps["rf"].classes_ if hasattr(pipe, "named_steps") and "rf" in pipe.named_steps else getattr(pipe, "classes_", None)
        probs = pipe.predict_proba(X_all)
        if classes is not None and 2 in classes:
            idx = list(classes).index(2)
            df_risk = meta_all.copy()
            df_risk["prob_fail"] = [p[idx] for p in probs]
            df_risk = df_risk.sort_values("prob_fail", ascending=False)
            # replace deprecated use_container_width with width="stretch"
            st.dataframe(df_risk.head(10), width="stretch")
        else:
            st.info("Model loaded but target class '2' not in model classes.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Model not loaded or not enough windows available.")

# -------------------------
# Footer (neutral, no mentions)
# -------------------------
st.markdown("---")

