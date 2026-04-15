import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Metro Crowd Density Estimator",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0a0f; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #111827 100%); }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value { font-size: 2.8rem; font-weight: 700; line-height: 1; }
    .metric-label {
        font-size: 0.78rem; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem;
    }
    .status-normal {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981; border-radius: 12px;
        padding: 1rem 1.5rem; color: #34d399;
        font-size: 1.3rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .status-moderate {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 1px solid #f59e0b; border-radius: 12px;
        padding: 1rem 1.5rem; color: #fbbf24;
        font-size: 1.3rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .status-crowded {
        background: linear-gradient(135deg, #7c2d12, #9a3412);
        border: 1px solid #f97316; border-radius: 12px;
        padding: 1rem 1.5rem; color: #fb923c;
        font-size: 1.3rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .status-overcrowded {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444; border-radius: 12px;
        padding: 1rem 1.5rem; color: #f87171;
        font-size: 1.3rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .section-header {
        color: #e5e7eb; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .info-box {
        background: rgba(96,165,250,0.08);
        border: 1px solid rgba(96,165,250,0.2);
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #93c5fd; font-size: 0.85rem; margin-top: 0.5rem;
    }
    .sidebar-metric {
        background: rgba(255,255,255,0.04);
        border-radius: 10px; padding: 0.6rem 0.8rem;
        margin: 0.3rem 0; font-size: 0.82rem; color: #d1d5db;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(96,165,250,0.3);
        border-radius: 16px; padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 600; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Model definition ──────────────────────────────────────────
def make_layers(cfg, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = make_layers(
            [64,64,'M',128,128,'M',256,256,256,'M',512,512,512])
        self.backend  = make_layers(
            [512,512,512,256,128,64], in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

@st.cache_resource
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load(
        'weights/csrnet_v3_best.pth', map_location='cpu'))
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────
def run_inference(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        density_map = model(tensor).squeeze().numpy()
    density_map = np.maximum(density_map, 0)   # clamp negatives
    count = max(0, int(density_map.sum()))      # never show negative
    return density_map, count

def get_status(count):
    if count < 30:
        return 'NORMAL',      '#10b981', 'status-normal',      '🟢'
    elif count < 70:
        return 'MODERATE',    '#f59e0b', 'status-moderate',    '🟡'
    elif count < 120:
        return 'CROWDED',     '#f97316', 'status-crowded',     '🟠'
    else:
        return 'OVERCROWDED', '#ef4444', 'status-overcrowded', '🔴'

def generate_heatmap(image: Image.Image, density_map, count):
    img_np = np.array(image.convert('RGB'))

    density_map     = np.maximum(density_map, 0)
    density_resized = cv2.resize(
        density_map, (img_np.shape[1], img_np.shape[0]))
    density_resized = np.maximum(density_resized, 0)

    actual_max = density_resized.max()

    if actual_max < 1e-5:
        # completely empty — all blue
        density_norm = np.zeros_like(
            density_resized, dtype=np.uint8)

    elif count < 15:
        # sparse — gentle normalization, mostly blue
        density_norm = np.clip(
            density_resized / actual_max * 80,
            0, 255).astype(np.uint8)

    elif count < 50:
        # moderate — medium normalization, warm patches
        density_norm = np.clip(
            density_resized / actual_max * 160,
            0, 255).astype(np.uint8)

    else:
        # dense — full normalization, red hotspots
        p1  = np.percentile(density_resized, 2)
        p99 = np.percentile(density_resized, 98)
        density_clipped = np.clip(density_resized, p1, p99)
        density_norm = ((density_clipped - p1) /
                        (p99 - p1 + 1e-8) * 255).astype(np.uint8)

    heatmap     = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)

    return density_norm, overlay

# ── UI ────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🚇 Metro Crowd Density Estimator</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">CSRNet-based real-time crowd density '
    'estimation for Indian public transport</div>',
    unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️ System Info')
    st.markdown('<div class="sidebar-metric">🧠 Model: CSRNet v3</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">📊 Metro MAE: 12.36</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">📈 Metro MSE: 126.44</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🏋️ Epochs: 50</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🖼️ Train images: 788</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">'
                '&nbsp;&nbsp;&nbsp;├ ShanghaiTech A: 300</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">'
                '&nbsp;&nbsp;&nbsp;├ ShanghaiTech B: 400</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">'
                '&nbsp;&nbsp;&nbsp;└ Indian Metro: 88</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">📍 Domain: Indian Metro</div>',
                unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### 🚦 Alert Thresholds')
    st.markdown('<div class="sidebar-metric">🟢 Normal: 0–29</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🟡 Moderate: 30–69</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🟠 Crowded: 70–119</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🔴 Overcrowded: 120+</div>',
                unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### 📋 Approach Comparison')
    st.markdown(
        '<div class="sidebar-metric">❌ CNN Classifier — MAE: ~55</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-metric">❌ YOLOv8 — MAE: 283.23</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-metric">✅ CSRNet v3 — MAE: 12.36</div>',
        unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### 👨‍💻 Project Info')
    st.markdown('''
    **Project #14** — AI/Computer Vision  
    **Student:** Abdur Rahman Qasim  
    **Guide:** Dr. Shivani Yadao  
    **Deadline:** April 15, 2026
    ''')

# ── Main content ──────────────────────────────────────────────
model = load_model()

uploaded = st.file_uploader(
    "Upload a metro/transport crowd image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload any crowd image from a metro station, "
         "bus terminal, or railway platform"
)

if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')

    with st.spinner('Analysing crowd density...'):
        density_map, count       = run_inference(image, model)
        density_norm, overlay    = generate_heatmap(
            image, density_map, count)
        status, color, css_class, emoji = get_status(count)

    # ── Status banner ─────────────────────────────────────────
    st.markdown(
        f'<div class="{css_class}">'
        f'{emoji} &nbsp; {status} &nbsp;|&nbsp; '
        f'{count} people detected</div>',
        unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">'
            f'{count}</div>'
            f'<div class="metric-label">People Detected</div>'
            f'</div>', unsafe_allow_html=True)
    with c2:
        density_pct = min(int(count / 200 * 100), 100)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">'
            f'{density_pct}%</div>'
            f'<div class="metric-label">Capacity Used</div>'
            f'</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#60a5fa">'
            f'12.36</div>'
            f'<div class="metric-label">Model MAE</div>'
            f'</div>', unsafe_allow_html=True)
    with c4:
        img_w, img_h = image.size
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" '
            f'style="color:#a78bfa;font-size:1.4rem">'
            f'{img_w}×{img_h}</div>'
            f'<div class="metric-label">Image Resolution</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Image panels ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📸 Analysis Output</div>',
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Original Image**')
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('**Density Map**')
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        im = ax.imshow(density_norm, cmap='jet')
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label='Crowd Density')
        ax.set_title('Heat intensity = crowd concentration',
                     color='#9ca3af', fontsize=8, pad=6)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown('**Heatmap Overlay**')
        st.image(overlay, use_container_width=True)

    # ── Density distribution chart ────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">'
        '📊 Density Distribution Analysis</div>',
        unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        flat     = density_map.flatten()
        flat_pos = flat[flat > 0.01]
        if len(flat_pos) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            fig2.patch.set_facecolor('#111827')
            ax2.set_facecolor('#1f2937')
            ax2.hist(flat_pos, bins=50, color='#60a5fa',
                     edgecolor='none', alpha=0.8)
            ax2.set_xlabel('Density Value', color='#9ca3af')
            ax2.set_ylabel('Pixel Count',   color='#9ca3af')
            ax2.set_title(
                'Distribution of Crowd Density Values',
                color='#e5e7eb', fontsize=10)
            ax2.tick_params(colors='#6b7280')
            for spine in ax2.spines.values():
                spine.set_color('#374151')
            st.pyplot(fig2)
            plt.close()
        else:
            st.info('No significant crowd density detected.')

    with col_b:
        st.markdown('<br>', unsafe_allow_html=True)
        peak_val = float(density_map.max())
        active   = int((density_map > 0.01).sum())
        mean_val = float(density_map[density_map > 0.01].mean()
                         if active > 0 else 0)
        st.markdown(
            f'<div class="info-box">'
            f'🔺 Peak density: <b>{peak_val:.3f}</b><br>'
            f'📊 Mean density: <b>{mean_val:.3f}</b><br>'
            f'🗺️ Active pixels: <b>{active}</b><br>'
            f'📐 Map size: <b>64×64</b>'
            f'</div>', unsafe_allow_html=True)

    # ── Recommendation ────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    if status == 'NORMAL':
        st.success(
            '✅ Platform is within safe capacity. '
            'No action required.')
    elif status == 'MODERATE':
        st.warning(
            '⚠️ Moderate crowd detected. '
            'Monitor the situation.')
    elif status == 'CROWDED':
        st.warning(
            '🟠 Platform is crowded. Consider '
            'crowd management measures.')
    else:
        st.error(
            '🚨 OVERCROWDING ALERT! Immediate crowd '
            'management required. Consider halting entry.')

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value" style="color:#60a5fa">788</div>
            <div class="metric-label">Training Images</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value" style="color:#34d399">12.36</div>
            <div class="metric-label">Metro MAE</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value" style="color:#a78bfa">50</div>
            <div class="metric-label">Training Epochs</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('''
    <div class="info-box" style="text-align:center; padding: 2rem;">
        👆 Upload a metro or transport crowd image above
        to get started<br><br>
        <span style="color:#6b7280; font-size:0.85rem">
        Supports JPG, JPEG, PNG • Any resolution
        </span>
    </div>
    ''', unsafe_allow_html=True)