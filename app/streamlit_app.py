"""
Streamlit UI — Predictive Maintenance Predictor
Huashu design · Industrial Engineering theme
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd

from src.predict import predict_single, load_type_encoder
from src.preprocess import TARGETS, TARGET_LABELS, RANDOM_STATE

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediktif Maintenance — Failure Predictor",
    page_icon="⚙",
    layout="centered",
)

# ── Custom CSS · Huashu Industrial Engineering ─────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Reset & Base ───────────────────────────────────────────── */
html, body, #root, .stApp, .stApp > header, main {
    background: #F6F5F2 !important;
}
.stApp {
    background: #F6F5F2 !important;
}
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E0DC;
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    padding: 0.625rem 1rem;
    border-radius: 6px;
    border: 1px solid #E2E0DC;
    background: #FFFFFF;
    color: #1B3A5C;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: #1B3A5C;
    background: #E8EDF2;
}
section[data-testid="stSidebar"] .stButton button:active {
    background: #D4DDE8;
}
section[data-testid="stSidebar"] hr {
    border-color: #E2E0DC;
    margin: 1.25rem 0;
}
section[data-testid="stSidebar"] .sidebar-content {
    padding: 1.5rem 1rem;
}

/* ── Typography ─────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #1A1A1A;
    font-weight: 600;
    letter-spacing: -0.02em;
}
h1 {
    font-size: 1.75rem;
    margin-bottom: 0.25rem;
}
h2 {
    font-size: 1.25rem;
    margin-bottom: 0.75rem;
}
p, li, .stMarkdown, .stCaption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #4A4A4A;
    line-height: 1.6;
}
.stCaption {
    font-size: 0.8rem;
    color: #8A8A8A;
}

/* ── Page Header ────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 2px solid #1B3A5C;
    margin-bottom: 1.5rem;
}
.app-header .header-icon {
    width: 42px;
    height: 42px;
    background: #1B3A5C;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.app-header .header-icon svg {
    width: 24px;
    height: 24px;
    fill: none;
    stroke: #FFFFFF;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.app-header .header-text h1 {
    margin: 0;
    font-size: 1.5rem;
    color: #1B3A5C;
}
.app-header .header-text p {
    margin: 0;
    font-size: 0.875rem;
    color: #6B7280;
}

/* ── Cards / Containers ─────────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E0DC;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #1B3A5C;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E2E0DC;
}

/* ── Input Section ──────────────────────────────────────────── */
/* Style the form container itself */
section[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E2E0DC;
    border-radius: 8px;
    padding: 1.25rem 1.5rem 0.75rem 1.5rem;
}
.param-risk {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 1px 8px;
    border-radius: 4px;
    margin-left: auto;
}
.param-risk.safe {
    color: #4A7C59;
    background: #EAF3EC;
}
.param-risk.warning {
    color: #B8860B;
    background: #FDF3D0;
}
.param-risk.danger {
    color: #C44A4A;
    background: #F9E6E6;
}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton button[kind="primary"] {
    background: #1B3A5C !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: background 0.15s ease !important;
}
.stButton button[kind="primary"]:hover {
    background: #2A4F74 !important;
}
.stButton button[kind="primary"]:active {
    background: #142C47 !important;
}

/* ── Status Banners ─────────────────────────────────────────── */
.status-banner {
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
    font-weight: 500;
}
.status-banner.danger {
    background: #F9E6E6;
    border-left-color: #C44A4A;
    color: #8B2D2D;
}
.status-banner.warning {
    background: #FDF3D0;
    border-left-color: #D4A030;
    color: #7A6200;
}
.status-banner.success {
    background: #EAF3EC;
    border-left-color: #4A7C59;
    color: #2D5A3A;
}
.status-banner .banner-stat {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

/* ── Divider ────────────────────────────────────────────────── */
hr, .stDivider {
    border-color: #E2E0DC !important;
    margin: 1.25rem 0 !important;
}

/* ── DataFrames ─────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #E2E0DC;
    border-radius: 6px;
    overflow: hidden;
}
.stDataFrame table {
    font-size: 0.85rem;
}

/* ── Expander ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    color: #1B3A5C !important;
    font-size: 0.875rem !important;
    border: 1px solid #E2E0DC !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    background: #FFFFFF !important;
}
.streamlit-expanderHeader:hover {
    background: #F3F2EF !important;
}
.streamlit-expanderContent {
    border: 1px solid #E2E0DC;
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 1rem !important;
    background: #FFFFFF;
}

/* ── Selectbox ──────────────────────────────────────────────── */
.stSelectbox label {
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #4A4A4A !important;
}
.stSelectbox div[data-baseweb="select"] {
    border-color: #D4D4D4 !important;
    border-radius: 6px !important;
}

/* ── Slider ─────────────────────────────────────────────────── */
.stSlider label {
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #4A4A4A !important;
}
div[data-testid="stTickBar"] {
    font-size: 0.7rem !important;
    color: #8A8A8A !important;
}

/* ── Section title ──────────────────────────────────────────── */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.5rem 0 0.75rem 0;
    padding: 0;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: #1B3A5C;
    border-radius: 2px;
}

/* ── Tooltip style ──────────────────────────────────────────── */
.sidebar-tip {
    font-size: 0.8rem;
    color: #6B7280;
    line-height: 1.7;
}
.sidebar-tip strong {
    color: #1B3A5C;
}

/* ── Caption / Footer ───────────────────────────────────────── */
.footer-bar {
    padding: 1rem 0;
    border-top: 1px solid #E2E0DC;
    margin-top: 2rem;
    text-align: center;
    font-size: 0.75rem;
    color: #8A8A8A;
}

/* ── Dark Mode · Industrial Warm ──────────────────────────── */
@media (prefers-color-scheme: dark) {
    html, body, #root, .stApp, .stApp > header, main {
        background: #1A1D22 !important;
    }
    .stApp {
        background: #1A1D22 !important;
    }
    section[data-testid="stSidebar"] {
        background: #22262B;
        border-right: 1px solid #373C43;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: #22262B;
        border-color: #373C43;
        color: #D4D2CC;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2D3238;
        border-color: #5A8FBD;
    }
    section[data-testid="stSidebar"] .stButton button:active {
        background: #383E45;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #373C43;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #E4E2DD;
    }
    p, li, .stMarkdown, .stCaption {
        color: #B0ADA8;
    }
    .stCaption {
        color: #7A7874;
    }

    .app-header {
        border-bottom-color: #5A8FBD;
    }
    .app-header .header-icon {
        background: #5A8FBD;
    }
    .app-header .header-text h1 {
        color: #E4E2DD;
    }
    .app-header .header-text p {
        color: #9C9A94;
    }

    .card,
    section[data-testid="stForm"] {
        background: #22262B;
        border-color: #373C43;
    }
    .card-title {
        color: #9CBDD8;
        border-bottom-color: #373C43;
    }

    .param-risk.safe {
        color: #7CBF8B;
        background: #1E2E22;
    }
    .param-risk.warning {
        color: #E8C44A;
        background: #2E2818;
    }
    .param-risk.danger {
        color: #E07070;
        background: #2E1E1E;
    }

    .stButton button[kind="primary"] {
        background: #5A8FBD !important;
        color: #FFFFFF !important;
    }
    .stButton button[kind="primary"]:hover {
        background: #6DA4D4 !important;
    }
    .stButton button[kind="primary"]:active {
        background: #4A7DA8 !important;
    }

    .status-banner.danger {
        background: #2E1E1E;
        border-left-color: #C44A4A;
        color: #E0A0A0;
    }
    .status-banner.warning {
        background: #2E2818;
        border-left-color: #D4A030;
        color: #E0C070;
    }
    .status-banner.success {
        background: #1E2E22;
        border-left-color: #4A7C59;
        color: #7CBF8B;
    }

    hr, .stDivider {
        border-color: #373C43 !important;
    }

    .stDataFrame {
        border-color: #373C43;
    }
    .stDataFrame table,
    .stDataFrame td {
        color: #D4D2CC !important;
        background: #22262B !important;
    }
    .stDataFrame th {
        color: #9C9A94 !important;
        background: #1A1D22 !important;
        border-bottom-color: #373C43 !important;
    }
    tr:hover td {
        background: #2D3238 !important;
    }

    .streamlit-expanderHeader {
        color: #9CBDD8 !important;
        border-color: #373C43 !important;
        background: #22262B !important;
    }
    .streamlit-expanderHeader:hover {
        background: #2D3238 !important;
    }
    .streamlit-expanderContent {
        border-color: #373C43 !important;
        background: #22262B !important;
    }

    .stSelectbox label {
        color: #B0ADA8 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-color: #373C43 !important;
    }
    .stSlider label {
        color: #B0ADA8 !important;
    }
    div[data-testid="stTickBar"] {
        color: #7A7874 !important;
    }

    .section-title {
        color: #9C9A94;
    }
    .section-title::before {
        background: #5A8FBD;
    }

    .footer-bar {
        border-top-color: #373C43;
        color: #7A7874;
    }

    code {
        background: #2D3238 !important;
        color: #D4D2CC !important;
    }
}
</style>
"""

HEADER_SVG = """<svg viewBox="0 0 24 24" width="24" height="24">
  <circle cx="12" cy="12" r="2.5"/>
  <path d="M5 12h3M16 12h3M12 5v3M12 16v3"/>
  <circle cx="12" cy="12" r="9" stroke-width="1.5" fill="none"/>
</svg>"""

# ── Constants ──────────────────────────────────────────────────────────────
FEATURE_INFO = {
    "type": {
        "label": "Tipe Produk",
        "help": "L = Large, M = Medium, H = High",
        "risk_tip": "Tipe H (High) cenderung memiliki risiko kegagalan lebih tinggi karena beban kerja lebih besar.",
        "safe": ["L", "M"],
        "danger": ["H"],
    },
    "air_temp": {
        "label": "Suhu Udara",
        "unit": "K",
        "help": "Suhu udara sekitar mesin",
        "risk_tip": "Suhu > 305 K meningkatkan risiko Heat Dissipation Failure (HDF). Suhu normal: 295–302 K.",
        "safe": (295, 302),
        "warning": (302, 307),
        "danger": (307, 310),
    },
    "proc_temp": {
        "label": "Suhu Proses",
        "unit": "K",
        "help": "Suhu proses pemotongan",
        "risk_tip": "Suhu proses > 315 K mengindikasikan potensi overheating. Normal: 305–312 K.",
        "safe": (305, 312),
        "warning": (312, 317),
        "danger": (317, 320),
    },
    "rot_speed": {
        "label": "Kecepatan Rotasi",
        "unit": "rpm",
        "help": "Kecepatan rotasi spindel",
        "risk_tip": "Kecepatan > 2500 rpm atau < 1200 rpm meningkatkan risiko Power Failure (PWF) dan Overstrain (OSF).",
        "safe": (1200, 2000),
        "warning": (2000, 2500),
        "danger": (1000, 1200, 2500, 3000),
    },
    "torque": {
        "label": "Torsi",
        "unit": "Nm",
        "help": "Torsi yang diterapkan oleh spindel",
        "risk_tip": "Torsi > 60 Nm meningkatkan risiko Overstrain Failure (OSF). Normal: 10–50 Nm.",
        "safe": (0, 50),
        "warning": (50, 70),
        "danger": (70, 100),
    },
    "tool_wear": {
        "label": "Keausan Alat",
        "unit": "min",
        "help": "Durasi pemakaian alat saat ini",
        "risk_tip": "Keausan alat > 200 menit meningkatkan drastis risiko Tool Wear Failure (TWF). Normal: 0–150 menit.",
        "safe": (0, 150),
        "warning": (150, 220),
        "danger": (220, 300),
    },
}


def get_risk_level(feature_key: str, value) -> str:
    """Return 'safe', 'warning', or 'danger'."""
    info = FEATURE_INFO[feature_key]
    if feature_key == "type":
        return "danger" if value in info["danger"] else "safe"
    r = info.get("danger")
    w = info.get("warning")
    if r and len(r) == 4:
        if value <= r[1] or value >= r[2]:
            return "danger"
    elif r and value >= r[0]:
        return "danger"
    if w and value >= w[0]:
        return "warning"
    return "safe"


RISK_LABELS = {"safe": "Aman", "warning": "Waspada", "danger": "Kritis"}
RISK_CLASS = {"safe": "safe", "warning": "warning", "danger": "danger"}

FAILURE_COLORS = {
    "TWF": "#C44A4A",
    "HDF": "#D4823A",
    "PWF": "#D4A030",
    "OSF": "#4A7C59",
    "RNF": "#5A7D9C",
}

SEVERITY_RISK = {
    "TWF": "Tinggi — kegagalan alat dapat merusak mesin",
    "HDF": "Sedang — panas berlebih menyebabkan penyimpangan proses",
    "PWF": "Tinggi — kegagalan daya menghentikan produksi",
    "OSF": "Sedang — beban berlebih melelahkan komponen",
    "RNF": "Rendah — acak, tidak dapat dijelaskan secara fisik",
}


# ── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    le = load_type_encoder()
    type_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
    type_names = {int(idx): label for idx, label in enumerate(le.classes_)}
    return type_mapping, type_names


type_info, type_names = load_models()

# ── Inject CSS ─────────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-header">
        <div class="header-icon">{HEADER_SVG}</div>
        <div class="header-text">
            <h1>Prediktif Maintenance</h1>
            <p>Mesin CNC milling · Prediksi 5 tipe kegagalan · Random Forest + SMOTE</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="color:#6B7280;font-size:0.9rem;margin-bottom:1.5rem;">'
    "Atur 6 parameter mesin untuk memprediksi probabilitas "
    "kegagalan. Gunakan slider di bawah untuk eksplorasi cepat.</p>",
    unsafe_allow_html=True,
)

# ── Sidebar: Quick Scenarios ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.25rem 0 0.5rem 0;">'
        '<p style="font-size:0.8rem;font-weight:600;color:#1B3A5C;text-transform:uppercase;'
        'letter-spacing:0.04em;margin-bottom:0.75rem;">Skenario Cepat</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    def _set_demo_mode(mode: str):
        values = {
            "extreme": ("H", 308.0, 318.0, 2900, 85.0, 280),
            "normal": ("L", 298.0, 308.0, 1500, 35.0, 20),
            "default": ("L", 300.0, 310.0, 1500, 40.0, 50),
        }
        t, a, p, r, o, w = values[mode]
        st.session_state["demo_mode"] = mode
        st.session_state["type_select"] = t
        st.session_state["slider_air"] = a
        st.session_state["slider_proc"] = p
        st.session_state["slider_rot"] = r
        st.session_state["slider_torque"] = o
        st.session_state["slider_wear"] = w
        st.rerun()

    if st.button("Skenario Ekstrem", width='stretch'):
        _set_demo_mode("extreme")

    if st.button("Mesin Normal", width='stretch'):
        _set_demo_mode("normal")

    if st.button("Reset Default", width='stretch'):
        _set_demo_mode("default")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.8rem;font-weight:600;color:#1B3A5C;text-transform:uppercase;'
        'letter-spacing:0.04em;margin-bottom:0.5rem;">Cara Membaca Hasil</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-tip">'
        "• <strong>&lt; 30%</strong> — Risiko Rendah<br>"
        "• <strong>30–50%</strong> — Waspada<br>"
        "• <strong>&gt; 50%</strong> — Potensi Gagal"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem;color:#8A8A8A;line-height:1.6;">'
        "<strong>Dataset:</strong> AI4I 2020 (UCI)<br>"
        "<strong>Model:</strong> RF + SMOTE · 5 klasifikasi biner<br>"
        "<strong>Fokus:</strong> Deteksi dini kegagalan"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Input Parameters Section ───────────────────────────────────────────────
st.markdown('<div class="section-title">Parameter Mesin</div>', unsafe_allow_html=True)

# Determine default values per demo mode
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = "default"

defaults = {
    "extreme": ("H", 308.0, 318.0, 2900, 85.0, 280),
    "normal": ("L", 298.0, 308.0, 1500, 35.0, 20),
    "default": ("L", 300.0, 310.0, 1500, 40.0, 50),
}
default_type, default_air, default_proc, default_rot, default_torque, default_wear = defaults[
    st.session_state["demo_mode"] if st.session_state["demo_mode"] in defaults else "default"
]


def on_slider_change():
    st.session_state["demo_mode"] = "custom"


with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        type_idx = st.selectbox(
            "Tipe Produk",
            options=list(type_names.values()),
            index=list(type_names.values()).index(default_type),
            help=FEATURE_INFO["type"]["help"],
            key="type_select",
        )

        air_temp = st.slider(
            "Suhu Udara",
            min_value=295.0,
            max_value=310.0,
            value=default_air,
            step=0.5,
            format="%.1f K",
            help=FEATURE_INFO["air_temp"]["risk_tip"],
            key="slider_air",
        )

        proc_temp = st.slider(
            "Suhu Proses",
            min_value=305.0,
            max_value=320.0,
            value=default_proc,
            step=0.5,
            format="%.1f K",
            help=FEATURE_INFO["proc_temp"]["risk_tip"],
            key="slider_proc",
        )

    with col2:
        rot_speed = st.slider(
            "Kecepatan Rotasi",
            min_value=1000,
            max_value=3000,
            value=default_rot,
            step=10,
            format="%d rpm",
            help=FEATURE_INFO["rot_speed"]["risk_tip"],
            key="slider_rot",
        )

        torque = st.slider(
            "Torsi",
            min_value=0.0,
            max_value=100.0,
            value=default_torque,
            step=1.0,
            format="%.1f Nm",
            help=FEATURE_INFO["torque"]["risk_tip"],
            key="slider_torque",
        )

        tool_wear = st.slider(
            "Keausan Alat",
            min_value=0,
            max_value=300,
            value=default_wear,
            step=5,
            format="%d min",
            help=FEATURE_INFO["tool_wear"]["risk_tip"],
            key="slider_wear",
        )

    # ── Inline Risk Indicators ─────────────────────────────────────────
    st.markdown(
        '<div style="margin: 0.25rem 0 1rem 0; display: flex; flex-wrap: wrap; gap: 0.5rem;">',
        unsafe_allow_html=True,
    )
    for key, val in [
        ("type", type_idx),
        ("air_temp", air_temp),
        ("proc_temp", proc_temp),
        ("rot_speed", rot_speed),
        ("torque", torque),
        ("tool_wear", tool_wear),
    ]:
        risk = get_risk_level(key, val)
        label = FEATURE_INFO[key]["label"]
        st.markdown(
            f'<span class="param-risk {RISK_CLASS[risk]}">'
            f"{RISK_LABELS[risk]} — {label}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit Button ──────────────────────────────────────────────────
    submitted = st.form_submit_button("Prediksi Kegagalan", type="primary", width='stretch')

# ── Results Section ────────────────────────────────────────────────────────
if submitted:
    with st.spinner("Running prediction..."):
        type_encoded = type_info[type_idx]
        results = predict_single(
            type_val=type_encoded,
            air_temp=air_temp,
            proc_temp=proc_temp,
            rot_speed=rot_speed,
            torque=torque,
            tool_wear=tool_wear,
        )

    st.session_state["last_results"] = results
    st.session_state["last_input"] = {
        "type": type_idx,
        "air_temp": air_temp,
        "proc_temp": proc_temp,
        "rot_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

if "last_results" in st.session_state:
    results = st.session_state["last_results"]

    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    any_failure = any(r["prediction"] == 1 for r in results.values())
    max_prob = max(r["probability"] for r in results.values())
    max_target = max(results, key=lambda t: results[t]["probability"])

    # Determine status
    if any_failure:
        status_class = "danger"
        status_icon = "●"
        status_title = "Potensi kegagalan terdeteksi"
        status_desc = f"Probabilitas tertinggi: <strong>{max_prob:.1%}</strong> ({max_target} — {TARGET_LABELS[max_target]})"
    elif max_prob > 0.3:
        status_class = "warning"
        status_icon = "●"
        status_title = "Perlu perhatian"
        status_desc = f"Probabilitas tertinggi: <strong>{max_prob:.1%}</strong> ({max_target} — {TARGET_LABELS[max_target]})"
    else:
        status_class = "success"
        status_icon = "●"
        status_title = "Mesin beroperasi normal"
        status_desc = f"Probabilitas kegagalan tertinggi: <strong>{max_prob:.1%}</strong>"

    st.markdown(
        f'<div class="status-banner {status_class}">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;">'
        f'<span style="font-size:1.2rem;">{status_icon}</span>'
        f'<span>{status_title}</span>'
        f"</div>"
        f'<div class="banner-stat">{status_desc}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Results table ─────────────────────────────────────────────────
    sorted_targets = sorted(TARGETS, key=lambda t: results[t]["probability"], reverse=True)

    result_data = []
    for rank, target in enumerate(sorted_targets, 1):
        res = results[target]
        prob = res["probability"]
        severity = SEVERITY_RISK[target]
        result_data.append(
            {
                "Kode": target,
                "Tipe Kegagalan": TARGET_LABELS[target],
                "Probabilitas": f"{prob:.1%}",
                "Severitas": severity,
            }
        )

    df_result = pd.DataFrame(result_data)

    def color_prob(val):
        try:
            p = float(val.strip("%")) / 100
            if p >= 0.5:
                return "color: #C44A4A; font-weight: 700"
            elif p >= 0.3:
                return "color: #B8860B; font-weight: 600"
            else:
                return "color: #4A7C59"
        except:
            return ""

    def color_code(val):
        if val in FAILURE_COLORS:
            return f"color: {FAILURE_COLORS[val]}; font-weight: 600"
        return ""

    styled_df = (
        df_result.style.map(color_prob, subset=["Probabilitas"])
        .map(color_code, subset=["Kode"])
        .set_table_styles(
            [
                {"selector": "th", "props": [("font-size", "0.8rem"), ("color", "#6B7280"), ("font-weight", "600"), ("text-transform", "uppercase"), ("letter-spacing", "0.03em"), ("border-bottom", "2px solid #E2E0DC")]},
                {"selector": "td", "props": [("font-size", "0.85rem"), ("padding", "0.5rem 0.75rem")]},
                {"selector": "tr:hover td", "props": [("background", "#F6F5F2")]},
            ]
        )
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Rincian Probabilitas Kegagalan</div>', unsafe_allow_html=True)
    st.dataframe(styled_df, width='stretch', hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Probability bar chart ─────────────────────────────────────────
    try:
        import altair as alt

        chart_data = pd.DataFrame(
            {
                "Failure Type": [f"{t} — {TARGET_LABELS[t]}" for t in TARGETS],
                "Code": [t for t in TARGETS],
                "Probability (%)": [results[t]["probability"] * 100 for t in TARGETS],
            }
        )

        sort_order = chart_data.sort_values("Probability (%)", ascending=False)[
            "Failure Type"
        ].tolist()

        bars = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, size=28)
            .encode(
                x=alt.X(
                    "Probability (%):Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title=None,
                    axis=alt.Axis(
                        grid=True, gridColor="#E8E6E2", gridOpacity=0.6,
                        tickCount=5, labelFontSize=11, labelColor="#6B7280",
                    ),
                ),
                y=alt.Y(
                    "Failure Type:N",
                    sort=sort_order,
                    title=None,
                    axis=alt.Axis(
                        labelFontSize=13, labelFontWeight=500,
                        labelPadding=8, labelColor="#4A4A4A",
                    ),
                ),
                color=alt.Color(
                    "Code:N",
                    scale=alt.Scale(
                        domain=list(FAILURE_COLORS.keys()),
                        range=list(FAILURE_COLORS.values()),
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Failure Type:N", title="Tipe"),
                    alt.Tooltip("Probability (%):Q", title="Risiko (%)", format=".1f"),
                ],
            )
        )

        labels = (
            alt.Chart(chart_data)
            .mark_text(align="left", dx=5, fontSize=13, fontWeight=600, color="#4A4A4A")
            .encode(
                x=alt.X("Probability (%):Q"),
                y=alt.Y("Failure Type:N", sort=sort_order, title=None),
                text=alt.Text("Probability (%):Q", format=".1f"),
            )
        )

        threshold_line = (
            alt.Chart(pd.DataFrame({"v": [50]}))
            .mark_rule(color="#C44A4A", strokeDash=[5, 3], size=1.5, opacity=0.45)
            .encode(x=alt.datum(50))
        )

        threshold_lbl = (
            alt.Chart(pd.DataFrame({"x": [50], "label": ["Ambang 50%"]}))
            .mark_text(align="left", dx=4, dy=-10, fontSize=11, color="#C44A4A")
            .encode(x=alt.datum(50), text="label:N")
        )

        chart = (
            (bars + labels + threshold_line + threshold_lbl)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFontSize=12, labelColor="#6B7280", gridDash=[2, 2])
            .properties(height=280)
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Probabilitas per Tipe Kegagalan</div>', unsafe_allow_html=True)
        st.altair_chart(chart, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    except ImportError:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fallback_data = pd.DataFrame(
            {
                "Failure Type": [f"{t}" for t in TARGETS],
                "Probability (%)": [results[t]["probability"] * 100 for t in TARGETS],
            }
        )
        st.bar_chart(fallback_data, x="Failure Type", y="Probability (%)", height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Interpretation ────────────────────────────────────────────────
    with st.expander("Interpretasi & Saran", expanded=True):
        all_inputs = [
            ("type", type_idx),
            ("air_temp", air_temp),
            ("proc_temp", proc_temp),
            ("rot_speed", rot_speed),
            ("torque", torque),
            ("tool_wear", tool_wear),
        ]

        risk_items = []
        for key, val in all_inputs:
            risk = get_risk_level(key, val)
            cls = RISK_CLASS[risk]
            lbl = RISK_LABELS[risk]
            tip = FEATURE_INFO[key]["risk_tip"]
            risk_items.append(
                f"<li>"
                f'<span class="param-risk {cls}" style="margin-left:0;">{lbl}</span> '
                f"<strong>{FEATURE_INFO[key]['label']}</strong> = {val}{' ' + FEATURE_INFO[key].get('unit', '')} — {tip}"
                f"</li>"
            )

        st.markdown(
            "<ul style='line-height:1.8;padding-left:1.25rem;margin:0;'>"
            + "".join(risk_items)
            + "</ul>",
            unsafe_allow_html=True,
        )

    # ── Input summary ─────────────────────────────────────────────────
    with st.expander("Ringkasan Input"):
        input_data = pd.DataFrame(
            {
                "Parameter": [
                    "Tipe Produk",
                    "Suhu Udara (K)",
                    "Suhu Proses (K)",
                    "Kecepatan Rotasi (rpm)",
                    "Torsi (Nm)",
                    "Keausan Alat (min)",
                ],
                "Nilai": [
                    str(type_idx),
                    f"{air_temp:.1f}",
                    f"{proc_temp:.1f}",
                    str(rot_speed),
                    f"{torque:.1f}",
                    str(tool_wear),
                ],
            }
        )
        st.dataframe(input_data, width='stretch', hide_index=True)

else:
    # ── First visit: About section ────────────────────────────────────
    st.markdown('<div class="section-title">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
            <div>
                <p style="font-weight:600;color:#1B3A5C;margin-bottom:0.5rem;">Pipeline</p>
                <code style="font-size:0.8rem;background:#F3F2EF;padding:0.4rem 0.6rem;border-radius:4px;display:inline-block;">
                StandardScaler → SMOTE → SelectKBest → RF
                </code>
                <ul style="font-size:0.85rem;color:#4A4A4A;padding-left:1.25rem;margin-top:0.75rem;line-height:1.7;">
                <li><strong>SMOTE</strong> — Synthetic sampling kelas minoritas</li>
                <li><strong>SelectKBest</strong> — Seleksi fitur</li>
                <li><strong>class_weight='balanced'</strong> — RF berbobot</li>
                <li><strong>Metrik</strong> — F1-macro, Recall, ROC-AUC</li>
                </ul>
            </div>
            <div>
                <p style="font-weight:600;color:#1B3A5C;margin-bottom:0.5rem;">Tipe Kegagalan</p>
                <div style="font-size:0.85rem;line-height:1.8;">
    """,
        unsafe_allow_html=True,
    )

    for code, label in TARGET_LABELS.items():
        color = FAILURE_COLORS.get(code, "#6B7280")
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
            f'background:{color};margin-right:0.4rem;"></span>'
            f"<strong>{code}</strong> — {label}<br>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
                </div>
            </div>
        </div>
        <div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid #E2E0DC;font-size:0.8rem;color:#8A8A8A;">
        <strong>Catatan:</strong> Dataset memiliki class imbalance ekstrem (99%+ normal).
        Pipeline dirancang untuk menghindari bias terhadap kelas normal.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "> Atur parameter mesin di atas dan klik **'Prediksi Kegagalan'** untuk melihat hasil."
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-bar">'
    "Dataset: AI4I 2020 (UCI) &nbsp;·&nbsp; "
    "Model: Random Forest + SMOTE &nbsp;·&nbsp; "
    "5 klasifikasi biner per tipe kegagalan"
    "</div>",
    unsafe_allow_html=True,
)
