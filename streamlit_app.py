"""
app.py — Crowd Density Estimation | Indian Metro
CSRNet Fine-tuned · MAE = 12.36

Two input modes:
  1. Sample Gallery  — picks from my_data/val/images/ automatically, no renaming needed
  2. Upload Your Own — drop in any image
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os
import glob

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd Density Estimator — Indian Metro",
    page_icon="🚇",
    layout="wide",
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "weights/csrnet_v3_best.pth"
VAL_DIR = "my_data/val/images"
SAMPLE_LABELS = [
    "Ameerpet — Peak Hour",
    "Raidurg — Platform",
    "Mumbai Central — Boarding",
    "Red Line — Pre-boarding",
    "Metro Corridor — Off-peak",
]

MAX_SAMPLES = 3  # max images to show in gallery

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from model import CSRNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# ── HELPERS ───────────────────────────────────────────────────────────────────
def predict(model, device, pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        density = model(tensor).squeeze().cpu().numpy()
    density = np.maximum(density, 0)
    return int(density.sum()), density


def get_alert(count):
    if count < 30:
        return "🟢 NORMAL", "#22c55e", "No action required"
    elif count < 70:
        return "🟡 MODERATE", "#eab308", "Monitor the situation"
    elif count < 120:
        return "🟠 CROWDED", "#f97316", "Deploy crowd management staff"
    else:
        return "🔴 OVERCROWDED", "#ef4444", "Halt platform entry immediately"


def normalise(density, count):
    if density.max() < 1e-8:
        return np.zeros_like(density, dtype=np.uint8)
    if count < 15:
        out = density / density.max() * 80
    elif count < 50:
        out = density / density.max() * 160
    else:
        p2, p98 = np.percentile(density, 2), np.percentile(density, 98)
        out = np.clip((density - p2) / (p98 - p2 + 1e-8), 0, 1) * 255
    return out.astype(np.uint8)


def make_heatmap(pil_img, density, count, alpha):
    orig = np.array(pil_img.convert("RGB"))
    h, w = orig.shape[:2]
    hmap = cv2.resize(density, (w, h))
    hmap = normalise(hmap, count)
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(blended)


def make_density_vis(density, size):
    hmap = cv2.resize(density, size)
    hmap = normalise(hmap, int(density.sum()))
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def get_sample_paths():
    """Dynamically load images from VAL_DIR."""
    if not os.path.exists(VAL_DIR):
        return []

    valid_exts = (".png", ".jpg", ".jpeg")
    all_images = sorted(
        [f for f in os.listdir(VAL_DIR) if f.lower().endswith(valid_exts)]
    )

    result = []
    for i, fname in enumerate(all_images[: MAX_SAMPLES * 3]):
        fpath = os.path.join(VAL_DIR, fname)
        label = SAMPLE_LABELS[i] if i < len(SAMPLE_LABELS) else fname
        result.append((label, fpath))
    return result


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 Crowd Density")
    st.caption("CSRNet Fine-tuned · Indian Metro · MAE = 12.36")
    st.markdown("---")
    alpha = st.slider("Heatmap opacity", 0.10, 0.90, 0.45, 0.05)
    st.markdown("---")
    st.markdown("**Alert thresholds**")
    for label, rng, color in [
        ("🟢 NORMAL", "< 30 people", "#22c55e"),
        ("🟡 MODERATE", "30 – 69 people", "#eab308"),
        ("🟠 CROWDED", "70 – 119 people", "#f97316"),
        ("🔴 OVERCROWDED", "≥ 120 people", "#ef4444"),
    ]:
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{label}</span>"
            f"<br><span style='font-size:12px;color:#94a3b8'>{rng}</span>",
            unsafe_allow_html=True,
        )
        st.write("")
    st.markdown("---")
    st.markdown("**Model checkpoint**")
    st.code("csrnet_v3_best.pth\nEpoch 14  ·  MAE = 11.30", language=None)
    st.markdown("---")
    st.caption("B.Tech CSE · 2025–26\nAbdur Rahman Qasim\nGuide: Dr. Shivani Yadao")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚇 Crowd Density Estimation — Indian Metro")
st.write(
    "Real-time crowd density estimation using CSRNet fine-tuned on Indian metro data. "
    "**Pick a sample image** from the gallery for an instant demo, or upload your own."
)
st.markdown("---")

# ── TWO-TAB INPUT ─────────────────────────────────────────────────────────────
tab_gallery, tab_upload = st.tabs(["📷  Sample Gallery", "⬆️  Upload Your Own"])

selected_pil = None
selected_name = None

# ══════════════════════════════════════
# TAB 1 — SAMPLE GALLERY
# ══════════════════════════════════════
with tab_gallery:
    sample_paths = get_sample_paths()

    if not sample_paths:
        st.warning(
            f"**No images found in `{VAL_DIR}/`.**\n\n"
            "Make sure your validation images are in that folder. "
            "Change `VAL_DIR` at the top of `app.py` if they are elsewhere."
        )
    else:
        st.markdown(
            f"**{len(sample_paths)} sample images** loaded from your validation set — "
            "click any to run the model instantly."
        )
        st.write("")

        cols = st.columns(3)

        for i, (label, fpath) in enumerate(sample_paths):
            with cols[i % 3]:
                try:
                    preview = Image.open(fpath).convert("RGB")
                    thumb = preview.copy()
                    thumb.thumbnail((300, 200))
                    st.image(thumb, use_container_width=True)
                except Exception:
                    st.error(f"Cannot read {fname}")
                    continue

                if st.button("▶  Run on this image", key=f"s_{i}"):
                    st.session_state["sel_path"] = fpath
                    st.session_state["sel_name"] = label
                    st.session_state.pop("up_img", None)
                    st.session_state.pop("up_name", None)

                st.caption(label)
                st.write("")

        if "sel_path" in st.session_state:
            try:
                selected_pil = Image.open(st.session_state["sel_path"]).convert("RGB")
                selected_name = st.session_state.get("sel_name", "Sample image")
                st.success(f"✅ Selected: **{selected_name}**")
            except Exception as e:
                st.error(f"Could not load: {e}")

# ══════════════════════════════════════
# TAB 2 — UPLOAD
# ══════════════════════════════════════
with tab_upload:
    st.markdown("Upload any metro platform or crowd image.")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded is not None:
        selected_pil = Image.open(uploaded).convert("RGB")
        selected_name = uploaded.name
        st.session_state.pop("sel_path", None)
        st.session_state.pop("sel_name", None)
        st.success(f"✅ Loaded: **{uploaded.name}**")

# ── INFERENCE + OUTPUT ────────────────────────────────────────────────────────
if selected_pil is not None:
    st.markdown("---")

    with st.spinner("Loading model…"):
        model, device = load_model()

    with st.spinner("Running CSRNet inference…"):
        count, density = predict(model, device, selected_pil)

    alert_label, alert_color, alert_action = get_alert(count)
    capacity = min(int(count / 200 * 100), 100)

    # Alert banner
    st.markdown(
        f"""
        <div style="
            background:{alert_color}15;border:2px solid {alert_color};
            border-radius:12px;padding:18px 28px;
            display:flex;align-items:center;gap:32px;margin-bottom:20px;">
            <div style="font-size:30px;font-weight:900;color:{alert_color};
                        white-space:nowrap">{alert_label}</div>
            <div>
                <div style="font-size:11px;color:#94a3b8;
                            text-transform:uppercase;letter-spacing:.08em">
                    Recommended action
                </div>
                <div style="font-size:17px;font-weight:600;color:#1e293b">
                    {alert_action}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated count", f"{count} people")
    k2.metric(
        "Platform capacity",
        f"{capacity}%",
        delta="Safe" if count < 70 else "⚠ At risk",
        delta_color="normal" if count < 70 else "inverse",
    )
    k3.metric("Alert level", alert_label.split(" ")[1])
    k4.metric("Device", str(device).upper())

    # Three-panel output
    st.markdown("#### Visual output")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Original**")
        st.image(selected_pil, use_container_width=True)

    with c2:
        st.markdown("**Density map**")
        st.image(make_density_vis(density, selected_pil.size), use_container_width=True)
        st.caption("🔵 Sparse → 🟢 Moderate → 🔴 Dense")

    with c3:
        st.markdown("**Heatmap overlay**")
        st.image(
            make_heatmap(selected_pil, density, count, alpha), use_container_width=True
        )
        st.caption(f"Opacity {alpha:.0%} · adjust in sidebar")

    # Extras
    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("📊 Density distribution", expanded=False):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 2.2))
            flat = density.flatten()
            flat = flat[flat > flat.max() * 0.02]
            ax.hist(flat, bins=40, color="#3b82f6", alpha=0.8, edgecolor="white")
            ax.set_xlabel("Density value", fontsize=9)
            ax.set_ylabel("Pixel count", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        with st.expander("📈 Model comparison", expanded=False):
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "Approach": "CNN Classifier",
                        "MAE": "~55",
                        "Density Map": "❌",
                        "Verdict": "Failed",
                    },
                    {
                        "Approach": "YOLOv8",
                        "MAE": "283.23",
                        "Density Map": "❌",
                        "Verdict": "Failed",
                    },
                    {
                        "Approach": "CSRNet Pretrained",
                        "MAE": "~50",
                        "Density Map": "✅",
                        "Verdict": "Baseline",
                    },
                    {
                        "Approach": "CSRNet Fine-tuned (this)",
                        "MAE": "12.36",
                        "Density Map": "✅",
                        "Verdict": "🏆 Best",
                    },
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "95.6% more accurate than YOLOv8 · 75.3% better than pretrained baseline"
            )

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
else:
    st.markdown(
        "### 👆 Select a sample above, or switch to Upload to use your own image"
    )
    st.write("")
    a1, a2, a3, a4 = st.columns(4)
    for col, (label, rng, color, action) in zip(
        [a1, a2, a3, a4],
        [
            ("🟢 NORMAL", "< 30", "#22c55e", "No action required"),
            ("🟡 MODERATE", "30–69", "#eab308", "Monitor situation"),
            ("🟠 CROWDED", "70–119", "#f97316", "Deploy staff"),
            ("🔴 OVERCROWDED", "≥ 120", "#ef4444", "Halt entry"),
        ],
    ):
        col.markdown(
            f"""<div style="background:{color}12;border:1.5px solid {color};
                border-radius:10px;padding:14px;text-align:center;">
                <div style="font-size:13px;font-weight:700;color:{color}">{label}</div>
                <div style="font-size:20px;font-weight:800;color:{color};margin:6px 0">{rng}</div>
                <div style="font-size:11px;color:#94a3b8">{action}</div>
            </div>""",
            unsafe_allow_html=True,
        )
