"""
============================================================
 streamlit_app.py — Smart Waste Detection Interactive Dashboard
============================================================
Upload an image → predict waste class → assign to bin →
update fill levels → view analytics. Run with:
    streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

from config import PATHS, CLASS_NAMES, NUM_CLASSES, BIN_MAPPING, FILL_CONTRIBUTION, AREA_THRESHOLDS
from utils.dataset_utils import get_device
from utils.training_utils import create_model
from utils.simulation_utils import predict_waste, estimate_fill_contribution, SmartDustbin

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="♻️ Smart Waste Detection System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2ECC71;
                   text-align: center; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1.1rem; color: #95a5a6; text-align: center;
                   margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 1.5rem; border-radius: 12px; color: white;
                   text-align: center; margin-bottom: 1rem; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.8; }
    .bin-full     { color: #e74c3c; font-weight: bold; }
    .bin-warning  { color: #f39c12; font-weight: bold; }
    .bin-ok       { color: #2ecc71; font-weight: bold; }
    .stAlert > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = PATHS["best_model"]

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model_tag = ckpt.get("model_tag", "resnet18")
        backbone = model_tag.split("_lr")[0]
        model = create_model(backbone, NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        val_acc = ckpt.get("best_val_acc", "N/A")
        info = f"✅ Loaded **{backbone}** (Val Acc: {val_acc}%)"
    else:
        model = create_model("resnet18", NUM_CLASSES, pretrained=True).to(device)
        backbone = "resnet18"
        info = "⚠️ No trained model found — using pretrained ResNet18 (not fine‑tuned)"

    model.eval()
    return model, device, backbone, info


# ── Session state ───────────────────────────────────────────
if "dustbin" not in st.session_state:
    st.session_state.dustbin = SmartDustbin(bin_mapping=BIN_MAPPING)
if "history" not in st.session_state:
    st.session_state.history = []

dustbin = st.session_state.dustbin


# ── Header ──────────────────────────────────────────────────
st.markdown('<div class="main-header">♻️ Smart Waste Detection System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI‑Powered Waste Classification, '
            'Segregation & Bin Fill Estimation</div>',
            unsafe_allow_html=True)

model, device, backbone, model_info = load_model()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(model_info)
    st.markdown(f"**Device:** `{device}`")
    st.markdown(f"**Classes:** {NUM_CLASSES}")
    st.markdown(f"**Model:** `{backbone}`")

    st.divider()
    if st.button("🗑️ Reset All Bins", type="primary", use_container_width=True):
        dustbin.reset_bins()
        st.session_state.history = []
        st.success("All bins emptied!")
        st.rerun()

    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    This is a **software‑only prototype** of a Smart AI Dustbin.
    - Fill estimation is approximate (image‑area heuristic)
    - Designed for later hardware extension
    - Built with PyTorch + Streamlit
    """)


# ── Main content ────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("📷 Upload Waste Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photo of waste to classify")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_result:
    st.subheader("🔍 Classification Result")

    if uploaded_file:
        with st.spinner("Classifying..."):
            # Predict
            pred = predict_waste(model, image, device, CLASS_NAMES)

            # Fill estimation
            size_config = {
                "thresholds": AREA_THRESHOLDS,
                "contributions": FILL_CONTRIBUTION,
            }
            fill_info = estimate_fill_contribution(image, size_config)

            # Update bin
            from utils.simulation_utils import map_class_to_bin
            bin_name = map_class_to_bin(pred["class_name"])

            event = dustbin.add_waste(
                class_name=pred["class_name"],
                confidence=pred["confidence"],
                size_category=fill_info["size_category"],
                fill_percent=fill_info["fill_percent"],
            )
            st.session_state.history.append(event)

        # Show results
        conf_color = "green" if pred["confidence"] > 0.8 else ("orange" if pred["confidence"] > 0.5 else "red")
        st.markdown(f"### Predicted: **{pred['class_name'].upper()}**")
        st.markdown(f"**Confidence:** :{conf_color}[{pred['confidence']:.1%}]")
        st.markdown(f"**Assigned Bin:** `{bin_name}`")
        st.markdown(f"**Estimated Size:** `{fill_info['size_category']}` "
                    f"(+{fill_info['fill_percent']}% fill)")

        # Probability bar chart
        probs = pred["probabilities"]
        fig_probs = go.Figure(go.Bar(
            y=list(probs.keys()),
            x=list(probs.values()),
            orientation='h',
            marker_color=['#2ECC71' if k == pred['class_name'] else '#3498db'
                          for k in probs.keys()],
        ))
        fig_probs.update_layout(
            title="Class Probabilities",
            xaxis_title="Probability",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    else:
        st.info("👆 Upload an image to see classification results.")

# ── Bin Status Dashboard ────────────────────────────────────
st.divider()
st.subheader("🗑️ Virtual Dustbin — Fill Status")

status = dustbin.get_status()
cols = st.columns(len(status))

BIN_COLORS = {
    "plastic_bin": "#3498db", "paper_bin": "#f39c12", "metal_bin": "#95a5a6",
    "glass_bin": "#2ecc71", "organic_bin": "#8B4513", "textile_bin": "#9b59b6",
    "hazardous_bin": "#e74c3c", "trash_bin": "#34495e",
}

for i, (name, info) in enumerate(status.items()):
    with cols[i]:
        fill = info["fill_percentage"]
        emoji = "🚨" if fill >= 80 else ("⚠️" if fill >= 50 else "✅")
        st.metric(
            label=f"{emoji} {name.replace('_', ' ').title()}",
            value=f"{fill:.0f}%",
            delta=f"{info['item_count']} items",
        )
        # Progress bar
        color = BIN_COLORS.get(name, "#777")
        st.progress(min(fill / 100, 1.0))

        if fill >= 100:
            st.error("🚨 FULL — Empty now!")
        elif fill >= 80:
            st.warning("⚠️ Nearing capacity!")

# ── Bin fill chart ──────────────────────────────────────────
if any(s["item_count"] > 0 for s in status.values()):
    st.divider()
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 Fill Level Chart")
        bin_names = list(status.keys())
        fill_values = [status[n]["fill_percentage"] for n in bin_names]
        colors = [BIN_COLORS.get(n, "#777") for n in bin_names]

        fig_fill = go.Figure(go.Bar(
            x=bin_names, y=fill_values,
            marker_color=colors,
            text=[f"{v:.0f}%" for v in fill_values],
            textposition="outside",
        ))
        fig_fill.add_hline(y=80, line_dash="dash", line_color="orange",
                           annotation_text="Warning")
        fig_fill.add_hline(y=100, line_dash="dash", line_color="red",
                           annotation_text="Full")
        fig_fill.update_layout(yaxis_range=[0, 120], height=400,
                               xaxis_tickangle=-45)
        st.plotly_chart(fig_fill, use_container_width=True)

    with col_chart2:
        st.subheader("🥧 Items by Bin")
        item_counts = {n: status[n]["item_count"] for n in bin_names
                       if status[n]["item_count"] > 0}
        if item_counts:
            fig_pie = px.pie(
                values=list(item_counts.values()),
                names=list(item_counts.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

# ── Event History ───────────────────────────────────────────
st.divider()
st.subheader("📜 Event History")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    display_cols = ["timestamp", "waste_class", "confidence", "target_bin",
                    "size_category", "fill_contribution", "bin_fill_after"]
    available_cols = [c for c in display_cols if c in df_hist.columns]
    st.dataframe(df_hist[available_cols].iloc[::-1], use_container_width=True,
                 height=300)
else:
    st.info("No events yet. Upload an image to start!")

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.85rem;">
    Smart Waste Detection System — Software Prototype<br>
    Fill estimation is approximate (image‑area heuristic).
    Real deployment would need depth sensors & bin calibration.
</div>
""", unsafe_allow_html=True)
