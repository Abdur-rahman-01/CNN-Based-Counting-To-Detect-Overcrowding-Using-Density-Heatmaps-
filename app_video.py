import streamlit as st
import tempfile
import os
import time

from video_processor import load_model, process_video

st.set_page_config(
    page_title="Crowd Density — Video Heatmap",
    page_icon="🎥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 CSRNet · Video Mode")
    st.caption("Fine-tuned on Indian Metro · MAE = 12.36")
    st.markdown("---")

    frame_skip = st.slider(
        "Frame skip",
        min_value=1, max_value=10, value=2,
        help=(
            "Process every Nth frame.\n\n"
            "1 = every frame (slow on CPU)\n"
            "2 = every 2nd frame (recommended)\n"
            "5 = fast preview"
        )
    )
    alpha = st.slider(
        "Heatmap opacity",
        min_value=0.10, max_value=0.90,
        value=0.45, step=0.05
    )

    st.markdown("---")
    st.markdown("**Alert thresholds**")
    st.markdown("🟢 **Normal** — count < 30")
    st.markdown("🟡 **Moderate** — 30 to 69")
    st.markdown("🟠 **Crowded** — 70 to 119")
    st.markdown("🔴 **Overcrowded** — 120 or more")
    st.markdown("---")
    st.info(
        "💡 On CPU, frame skip 2–3 gives the best balance "
        "of speed and visual smoothness."
    )

# ── Main page ─────────────────────────────────────────────────────
st.title("🎥 Crowd Density Video Heatmap")
st.write(
    "Upload a metro or crowd video. "
    "The same video is returned with a **density heatmap and "
    "overcrowding alert** overlaid on every frame."
)

uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("👆 Upload a video to get started.")
    st.stop()

# Show original video immediately on upload
col_orig, col_heat = st.columns(2)
with col_orig:
    st.subheader("📹 Original video")
    st.video(uploaded)

# ── Load model (cached — only loads once per session) ─────────────
@st.cache_resource
def get_model():
    with st.spinner("Loading CSRNet model…"):
        return load_model("weights/csrnet_v3_best.pth")

model, device = get_model()
st.sidebar.success(f"✅ Model loaded · device: {device}")

# ── Save uploaded file to a temp location ─────────────────────────
ext = os.path.splitext(uploaded.name)[-1] or ".mp4"

tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
tmp_in.write(uploaded.read())
tmp_in.flush()
tmp_in.close()

input_path  = tmp_in.name
output_path = input_path.replace(ext, "_heatmap.mp4")

# ── Process ───────────────────────────────────────────────────────
st.markdown("---")
progress_bar = st.progress(0, text="Starting…")
t_start      = time.time()

def on_progress(frac):
    elapsed = time.time() - t_start
    eta     = (elapsed / frac) * (1 - frac) if frac > 0.01 else 0
    progress_bar.progress(
        min(frac, 1.0),
        text=f"Processing… {frac * 100:.0f}%  —  ETA {eta:.0f}s"
    )

try:
    stats = process_video(
        model, device,
        input_path  = input_path,
        output_path = output_path,
        frame_skip  = frame_skip,
        alpha       = alpha,
        progress_callback = on_progress,
    )
except Exception as e:
    st.error(f"Processing failed: {e}")
    os.unlink(input_path)
    st.stop()

elapsed = time.time() - t_start
progress_bar.progress(1.0, text="Done ✅")

# ── Stats ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg crowd count",  stats["avg_count"])
m2.metric("Peak crowd count", stats["peak_count"])
m3.metric("Total frames",     stats["total_frames"])
m4.metric("Processing time",  f"{elapsed:.1f}s")

# ── Show heatmap video ────────────────────────────────────────────
with col_heat:
    st.subheader("🔥 Heatmap output")
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

# ── Download ──────────────────────────────────────────────────────
st.download_button(
    label="⬇️  Download heatmap video",
    data=video_bytes,
    file_name=f"heatmap_{uploaded.name.replace(ext, '.mp4')}",
    mime="video/mp4",
    use_container_width=True,
)

# cleanup temp files
os.unlink(input_path)
os.unlink(output_path)