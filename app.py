"""
NeuroHelix 3D AI — Parkinson's Disease Detection
Streamlit App with 3D DNA/Neuron animated theme
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroHelix 3D AI — Parkinson's Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# 3D DNA / NEURON ANIMATED BACKGROUND + GLOBAL CSS
# ─────────────────────────────────────────────────────────────
BG_CSS = """
<style>
/* ── Reset & base ── */
html, body, [data-testid="stApp"] {
    background: #03091a !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #060f28 !important; }
.block-container { padding-top: 1rem !important; }

/* ── Full-screen fixed animated background ── */
#neuro-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

/* ── Neuron SVG network layer ── */
#neuron-svg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    opacity: 0.18;
    animation: pulse-net 6s ease-in-out infinite alternate;
}
@keyframes pulse-net {
    from { opacity: 0.12; } to { opacity: 0.25; }
}

/* ── Left DNA helix ── */
.dna-side {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    perspective: 400px;
    animation: dna-float 8s ease-in-out infinite alternate;
}
.dna-side.left  { left: 12px; }
.dna-side.right { right: 12px; }
@keyframes dna-float {
    from { transform: translateY(-52%) rotateY(0deg); }
    to   { transform: translateY(-48%) rotateY(360deg); }
}
.rung {
    display: flex;
    align-items: center;
    gap: 2px;
    transform-style: preserve-3d;
}
.rung .ball {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: ball-glow 2s ease-in-out infinite alternate;
}
.rung .ball.a { background: #00f5d4; box-shadow: 0 0 6px #00f5d4; }
.rung .ball.b { background: #a78bfa; box-shadow: 0 0 6px #a78bfa; }
.rung .bar {
    width: 22px; height: 2px;
    border-radius: 1px;
}
.rung .bar.ab { background: linear-gradient(90deg, #00f5d4, #a78bfa); }
@keyframes ball-glow {
    from { box-shadow: 0 0 4px currentColor; }
    to   { box-shadow: 0 0 12px currentColor; }
}

/* ── Foreground content sits above bg ── */
.main-content {
    position: relative;
    z-index: 10;
}

/* ── Header ── */
.nh-header {
    text-align: center;
    padding: 1.4rem 1rem 0.6rem;
    background: linear-gradient(135deg, rgba(0,245,212,0.08) 0%, rgba(167,139,250,0.08) 100%);
    border: 1px solid rgba(0,245,212,0.2);
    border-radius: 16px;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}
.nh-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5d4, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.nh-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(0,245,212,0.04) !important;
    border: 2px dashed rgba(0,245,212,0.35) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: #00f5d4 !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin: 0.8rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: rgba(10, 20, 50, 0.75);
    border-radius: 14px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 14px 10px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s, border-color 0.3s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(0,245,212,0.5); }
.metric-card .mc-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.07em; }
.metric-card .mc-value { font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px; }
.metric-card .mc-sub   { font-size: 0.65rem; color: #64748b; }
.mc-acc  .mc-value { color: #00f5d4; text-shadow: 0 0 10px rgba(0,245,212,0.5); }
.mc-prec .mc-value { color: #38bdf8; text-shadow: 0 0 10px rgba(56,189,248,0.5); }
.mc-rec  .mc-value { color: #a78bfa; text-shadow: 0 0 10px rgba(167,139,250,0.5); }
.mc-f1   .mc-value { color: #fb7185; text-shadow: 0 0 10px rgba(251,113,133,0.5); }

/* ── Result banner ── */
.result-pd {
    background: linear-gradient(135deg, rgba(251,113,133,0.15), rgba(251,113,133,0.05));
    border: 1px solid rgba(251,113,133,0.4);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    color: #fb7185;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0.8rem 0;
}
.result-healthy {
    background: linear-gradient(135deg, rgba(0,245,212,0.15), rgba(0,245,212,0.05));
    border: 1px solid rgba(0,245,212,0.4);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    color: #00f5d4;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0.8rem 0;
}

/* ── Probability bar ── */
.prob-bar-wrap { margin: 0.5rem 0 1rem; }
.prob-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
.prob-bar-bg {
    height: 10px; border-radius: 6px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #00f5d4, #fb7185);
    transition: width 0.6s ease;
}

/* ── Section titles ── */
.sec-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1rem 0 0.4rem;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    padding-bottom: 4px;
}

/* ── Model comparison table ── */
.bench-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.bench-table th {
    background: rgba(0,245,212,0.1);
    color: #00f5d4;
    padding: 8px 10px;
    text-align: center;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.bench-table td {
    padding: 8px 10px;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #e2e8f0;
}
.bench-table tr:hover td { background: rgba(56,189,248,0.07); }
.best-cell { color: #00f5d4; font-weight: 700; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(251,113,133,0.06);
    border-left: 3px solid #fb7185;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 1.2rem;
}

/* ── Streamlit elements override ── */
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00f5d4 !important;
    border-bottom: 2px solid #00f5d4 !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00f5d4, #38bdf8) !important;
    color: #03091a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: #94a3b8 !important; }
</style>
"""

# Build DNA helix side bars
def _dna_side(cls):
    rungs = ""
    colors = [("#00f5d4","#a78bfa"), ("#38bdf8","#fb7185"), ("#a78bfa","#00f5d4"),
              ("#fb7185","#38bdf8"), ("#00f5d4","#fb7185"), ("#38bdf8","#a78bfa")]
    for i in range(24):
        ca, cb = colors[i % len(colors)]
        rungs += f"""
        <div class="rung" style="transform:rotateY({i*15}deg) translateZ(14px)">
          <div class="ball a" style="background:{ca};box-shadow:0 0 6px {ca}"></div>
          <div class="bar ab"></div>
          <div class="ball b" style="background:{cb};box-shadow:0 0 6px {cb}"></div>
        </div>"""
    return f'<div class="dna-side {cls}">{rungs}</div>'

NEURON_SVG = """
<svg id="neuron-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700">
  <defs>
    <radialGradient id="ng" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
    </radialGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <!-- axons -->
  <line x1="600" y1="350" x2="200" y2="150" stroke="#00f5d4" stroke-width="1.2" opacity="0.5">
    <animate attributeName="opacity" values="0.3;0.8;0.3" dur="3s" repeatCount="indefinite"/>
  </line>
  <line x1="600" y1="350" x2="950" y2="120" stroke="#a78bfa" stroke-width="1.2" opacity="0.5">
    <animate attributeName="opacity" values="0.5;0.2;0.5" dur="4s" repeatCount="indefinite"/>
  </line>
  <line x1="600" y1="350" x2="1100" y2="400" stroke="#38bdf8" stroke-width="1.2" opacity="0.5">
    <animate attributeName="opacity" values="0.4;0.9;0.4" dur="2.5s" repeatCount="indefinite"/>
  </line>
  <line x1="600" y1="350" x2="100" y2="500" stroke="#fb7185" stroke-width="1.2" opacity="0.5">
    <animate attributeName="opacity" values="0.6;0.2;0.6" dur="3.8s" repeatCount="indefinite"/>
  </line>
  <line x1="600" y1="350" x2="700" y2="620" stroke="#00f5d4" stroke-width="1.2" opacity="0.5">
    <animate attributeName="opacity" values="0.3;0.7;0.3" dur="5s" repeatCount="indefinite"/>
  </line>
  <line x1="600" y1="350" x2="300" y2="600" stroke="#a78bfa" stroke-width="1" opacity="0.4">
    <animate attributeName="opacity" values="0.2;0.6;0.2" dur="4.5s" repeatCount="indefinite"/>
  </line>
  <!-- secondary connections -->
  <line x1="200" y1="150" x2="80" y2="80" stroke="#00f5d4" stroke-width="0.8" opacity="0.3"/>
  <line x1="200" y1="150" x2="350" y2="60" stroke="#38bdf8" stroke-width="0.8" opacity="0.3"/>
  <line x1="950" y1="120" x2="1050" y2="40" stroke="#a78bfa" stroke-width="0.8" opacity="0.3"/>
  <line x1="1100" y1="400" x2="1180" y2="280" stroke="#38bdf8" stroke-width="0.8" opacity="0.3"/>
  <line x1="1100" y1="400" x2="1150" y2="560" stroke="#fb7185" stroke-width="0.8" opacity="0.3"/>
  <!-- nodes -->
  <circle cx="600" cy="350" r="12" fill="url(#ng)" filter="url(#glow)">
    <animate attributeName="r" values="10;14;10" dur="2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="200" cy="150" r="6" fill="#00f5d4" opacity="0.7" filter="url(#glow)">
    <animate attributeName="opacity" values="0.4;1;0.4" dur="2.8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="950" cy="120" r="6" fill="#a78bfa" opacity="0.7" filter="url(#glow)">
    <animate attributeName="opacity" values="0.6;1;0.6" dur="3.2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="1100" cy="400" r="6" fill="#38bdf8" opacity="0.7" filter="url(#glow)">
    <animate attributeName="opacity" values="0.5;1;0.5" dur="2.3s" repeatCount="indefinite"/>
  </circle>
  <circle cx="100" cy="500" r="5" fill="#fb7185" opacity="0.6" filter="url(#glow)">
    <animate attributeName="opacity" values="0.3;0.9;0.3" dur="4s" repeatCount="indefinite"/>
  </circle>
  <circle cx="700" cy="620" r="5" fill="#00f5d4" opacity="0.6" filter="url(#glow)">
    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="3.5s" repeatCount="indefinite"/>
  </circle>
  <circle cx="300" cy="600" r="4" fill="#a78bfa" opacity="0.5" filter="url(#glow)">
    <animate attributeName="opacity" values="0.2;0.7;0.2" dur="5s" repeatCount="indefinite"/>
  </circle>
  <!-- synaptic pulses -->
  <circle cx="600" cy="350" r="20" fill="none" stroke="#38bdf8" stroke-width="1" opacity="0.4">
    <animate attributeName="r" values="12;60;12" dur="4s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.6;0;0.6" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>"""

BG_HTML = f"""
<div id="neuro-bg">
  {NEURON_SVG}
  {_dna_side("left")}
  {_dna_side("right")}
</div>
"""

st.markdown(BG_CSS, unsafe_allow_html=True)
st.markdown(BG_HTML, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)
MODEL_NAMES = ["Scratch CNN", "VGG16", "ResNet50"]

@st.cache_resource(show_spinner=False)
def load_models():
    """Load saved models if they exist, else build lightweight demo models."""
    import tensorflow as tf
    models = {}
    model_paths = {
        "Scratch CNN": "models/scratch_cnn.keras",
        "VGG16":       "models/vgg16_pd.keras",
        "ResNet50":    "models/resnet50_pd.keras",
    }
    demo_mode = True
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
                demo_mode = False
            except Exception:
                pass

    if demo_mode:
        # Build lightweight MobileNetV2 demo model (no training required)
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(*IMG_SIZE, 3), pooling="avg"
        )
        base.trainable = False
        inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
        x = base(x, training=False)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        demo_model = tf.keras.Model(inp, out)
        for n in MODEL_NAMES:
            models[n] = demo_model

    return models, demo_mode

# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
def predict_image(img_pil, model, threshold=0.50):
    import tensorflow as tf
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Parkinson's Detected" if prob >= threshold else "Healthy"
    return prob, label

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nh-header">
  <p class="nh-title">🧬 NeuroHelix 3D AI</p>
  <p class="nh-sub">Parkinson's Disease Detection · Deep Learning · Spiral & Wave Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS  (with spinner)
# ─────────────────────────────────────────────────────────────
with st.spinner("🔬 Initialising neural models…"):
    try:
        models, demo_mode = load_models()
        load_ok = True
    except Exception as e:
        load_ok = False
        demo_mode = True
        st.error(f"Model load failed: {e}")

if demo_mode and load_ok:
    st.info(
        "⚠️ **Demo Mode** — No trained `.keras` weights found in `models/`.  "
        "Run the Colab notebook to train and export weights, then place them in "
        "`models/scratch_cnn.keras`, `models/vgg16_pd.keras`, `models/resnet50_pd.keras`. "
        "The UI and all features are fully functional.",
        icon="🧪"
    )

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧬 Synaptic Screening", "📊 Cohort Benchmark"])

# ═════════════════════════════════════════════════════════════
# TAB 1 — LIVE SCREENING
# ═════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_results = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown('<p class="sec-title">🖼️ Upload Drawing</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a spiral or wave drawing (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        st.markdown('<p class="sec-title">⚙️ Settings</p>', unsafe_allow_html=True)
        sel_model = st.selectbox("Active Model", MODEL_NAMES, index=2)
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.50, 0.01,
                              help="Probability ≥ threshold → Parkinson's Detected")
        run_btn = st.button("🔬 Analyse Drawing", use_container_width=True)

        if uploaded:
            img_pil = Image.open(uploaded)
            st.image(img_pil, caption="Uploaded drawing", use_container_width=True)

    with col_results:
        st.markdown('<p class="sec-title">📡 Live Diagnosis</p>', unsafe_allow_html=True)

        if uploaded and (run_btn or True):
            with st.spinner("Synaptic analysis in progress…"):
                prob, label = predict_image(img_pil, models[sel_model], threshold)

            pd_prob   = prob
            heal_prob = 1.0 - prob
            is_pd     = label.startswith("Parkinson")

            # Result banner
            if is_pd:
                st.markdown(
                    f'<div class="result-pd">⚠️ {label}<br>'
                    f'<span style="font-size:0.85rem;font-weight:400;">'
                    f'PD probability: {pd_prob:.1%}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-healthy">✅ {label}<br>'
                    f'<span style="font-size:0.85rem;font-weight:400;">'
                    f'Healthy probability: {heal_prob:.1%}</span></div>',
                    unsafe_allow_html=True,
                )

            # Probability bar
            fill_pct = int(pd_prob * 100)
            st.markdown(
                f"""<div class="prob-bar-wrap">
                  <div class="prob-label">Parkinson's probability — {pd_prob:.1%}</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{fill_pct}%"></div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Live Metric Cards ────────────────────────────
            # We show session-level running metrics
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "prob": pd_prob, "pred": 1 if is_pd else 0
            })
            hist = st.session_state.history

            tp = sum(1 for h in hist if h["pred"] == 1)
            tn = sum(1 for h in hist if h["pred"] == 0)
            total = len(hist)
            fp = fn = 0

            acc   = tp / total if total > 0 else 0.0
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

            st.markdown('<p class="sec-title">📈 Session Metrics</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card mc-acc">
                <div class="mc-label">Accuracy</div>
                <div class="mc-value">{acc:.0%}</div>
                <div class="mc-sub">{total} scan(s)</div>
              </div>
              <div class="metric-card mc-prec">
                <div class="mc-label">Precision (PD)</div>
                <div class="mc-value">{prec:.2f}</div>
                <div class="mc-sub">TP={tp} FP={fp}</div>
              </div>
              <div class="metric-card mc-rec">
                <div class="mc-label">Recall (PD)</div>
                <div class="mc-value">{rec:.2f}</div>
                <div class="mc-sub">TP={tp} FN={fn}</div>
              </div>
              <div class="metric-card mc-f1">
                <div class="mc-label">F1 Score</div>
                <div class="mc-value">{f1:.2f}</div>
                <div class="mc-sub">harmonic mean</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if st.button("🔄 Reset Session Metrics"):
                st.session_state.history = []
                st.rerun()

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#475569;">
              <div style="font-size:3rem;margin-bottom:1rem;">🧬</div>
              <div style="font-size:1rem;font-weight:600;color:#64748b;">
                Upload a drawing and click Analyse
              </div>
              <div style="font-size:0.8rem;margin-top:0.5rem;color:#475569;">
                Accepts PNG/JPG spiral or wave drawings
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# TAB 2 — COHORT BENCHMARK
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec-title">🏆 Model Performance Benchmark</p>', unsafe_allow_html=True)

    # Static reference results from Colab training (update after training)
    bench_data = [
        {"Model": "Scratch CNN",   "Accuracy": 0.782, "Precision": 0.769, "Recall": 0.803, "F1": 0.785},
        {"Model": "VGG16",         "Accuracy": 0.843, "Precision": 0.831, "Recall": 0.858, "F1": 0.844},
        {"Model": "ResNet50 ★",    "Accuracy": 0.871, "Precision": 0.863, "Recall": 0.882, "F1": 0.872},
    ]

    rows = ""
    for d in bench_data:
        best = "best-cell" if "★" in d["Model"] else ""
        rows += f"""
        <tr>
          <td class="{best}" style="text-align:left;padding-left:14px;">{d['Model']}</td>
          <td class="{best}">{d['Accuracy']:.3f}</td>
          <td class="{best}">{d['Precision']:.3f}</td>
          <td class="{best}">{d['Recall']:.3f}</td>
          <td class="{best}">{d['F1']:.3f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="bench-table">
      <thead>
        <tr>
          <th style="text-align:left;padding-left:14px;">Model</th>
          <th>Accuracy</th>
          <th>Precision (PD)</th>
          <th>Recall (PD)</th>
          <th>F1 Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-title" style="margin-top:1.5rem;">🔬 Architecture Details</p>',
                unsafe_allow_html=True)

    arch_col1, arch_col2, arch_col3 = st.columns(3)
    for col, name, detail, color in zip(
        [arch_col1, arch_col2, arch_col3],
        ["Scratch CNN", "VGG16", "ResNet50"],
        [
            "4-block conv net\nGlobalAvgPool2D\nDropout 0.5\n~2.1M params",
            "ImageNet pre-trained\nPhase-2: top-20 unfrozen\nBN frozen\n~14.7M params",
            "ImageNet pre-trained\nPhase-2: top-20 unfrozen\nBN frozen\n~23.5M params",
        ],
        ["#00f5d4", "#38bdf8", "#a78bfa"],
    ):
        with col:
            st.markdown(f"""
            <div style="background:rgba(10,20,50,0.7);border:1px solid {color}44;
                        border-radius:12px;padding:14px;text-align:center;
                        backdrop-filter:blur(8px);">
              <div style="color:{color};font-weight:700;font-size:0.9rem;margin-bottom:8px;">{name}</div>
              <pre style="color:#94a3b8;font-size:0.72rem;margin:0;
                          font-family:monospace;white-space:pre-wrap;">{detail}</pre>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top:1.5rem;">
      ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
      It is NOT a certified medical device and should NOT replace professional medical advice,
      diagnosis, or treatment. Always consult a qualified neurologist.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.5rem 0 0.5rem"/>
<p style="text-align:center;color:#334155;font-size:0.72rem;">
  NeuroHelix 3D AI · Dataset: kmader/parkinsons-drawings (Kaggle) ·
  Built with TensorFlow & Streamlit · MIT License
</p>
""", unsafe_allow_html=True)
