# 🧬 NeuroHelix 3D AI — Parkinson's Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.14+-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  <a href="https://parkinson-disease-detection-ukrlka3vibzupvnoosnk3m.streamlit.app" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="35"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/mohansai2007/parkinson-disease-detection/blob/main/Parkinson_Detection_Kaggle_Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="35"/>
  </a>
</p>

<p align="center">
  <b>🚀 <a href="https://parkinson-disease-detection-bim9lnopggn3uf7bcendgx.streamlit.app/">👉 Click here for Live Demo</a> 🚀</b>
</p>

> **Early-stage Parkinson's Disease detection from spiral & wave drawings using Deep Learning — with a stunning 3D DNA/Neuron animated UI, live metrics (Accuracy · Precision · Recall · F1) and Grad-CAM explainability.**

---

## 🌐 Live Demo

**Try it instantly — no installation needed:**

> 🔗 **[https://parkinson-disease-detection-ukrlka3vibzupvnoosnk3m.streamlit.app](https://parkinson-disease-detection-bim9lnopggn3uf7bcendgx.streamlit.app/)**

Upload any spiral or wave drawing → get instant Parkinson's risk prediction with live metrics.

---

## 🎯 Project Overview

Parkinson's Disease (PD) affects motor control, which can be observed in hand-drawn spirals and waves. This project trains three deep learning models on the [`kmader/parkinsons-drawings`](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) Kaggle dataset and deploys them behind a full 3D animated medical AI dashboard.

### Key Features
- 🔬 **3 Models**: Scratch CNN · VGG16 · ResNet50 (two-phase transfer learning)
- 📊 **Full Metrics**: Accuracy · Precision · Recall · F1 (all Parkinson-class specific, `pos_label=1`)
- 🧬 **3D Animated UI**: Full-screen DNA helix + synaptic neuron network background (pure CSS 3D)
- 🗺️ **Grad-CAM**: Visual explainability heatmaps on misclassified samples
- ⚖️ **Youden's J Threshold**: Auto-calibrated optimal classification threshold
- 🧹 **Data Integrity**: MD5 exact + DCT pHash (Hamming ≤ 4) leakage removal before splitting
- 📈 **EDA**: Class distribution, pixel intensity analysis, Cohen's d, Welch's t-test

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| Source | [kmader/parkinsons-drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) on Kaggle |
| Download | Auto via `kagglehub` — no manual upload needed |
| Classes | `healthy` vs `parkinson` |
| Drawing types | Spiral · Wave |
| Splits | Train / Val (auto 18% carve) / Test |

---

## 🏗️ Model Architecture

| Model | Description | Trainable Params |
|-------|-------------|-----------------|
| **Scratch CNN** | Custom 4-block conv net with GlobalAvgPool + Dropout 0.5 | ~2.1M |
| **VGG16** | ImageNet pre-trained, Phase-2: top-20 layers unfrozen, BN frozen | ~14.7M |
| **ResNet50** | ImageNet pre-trained, Phase-2: top-20 layers unfrozen, BN frozen | ~23.5M |

---

## 📊 Results (Example — actual results vary by run)

| Model | Accuracy | Precision (PD) | Recall (PD) | F1 (PD) |
|-------|----------|----------------|-------------|---------|
| Scratch CNN | ~78% | ~0.76 | ~0.80 | ~0.78 |
| VGG16 | ~84% | ~0.83 | ~0.86 | ~0.84 |
| ResNet50 | ~87% | ~0.86 | ~0.88 | ~0.87 |

> Results depend on GPU, random seed, and Kaggle dataset version.

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended — Free GPU)

1. Click the **Open in Colab** badge above
2. Go to **Runtime → Change runtime type → GPU** (T4 recommended)
3. Add your Kaggle API key:
   - Colab sidebar → 🔑 Secrets → `KAGGLE_USERNAME` + `KAGGLE_KEY`
   - Or upload `~/.kaggle/kaggle.json`
4. **Run All** (`Ctrl+F9`)

### Option 2: Local (requires GPU recommended)

```bash
git clone https://github.com/mohansai2007/parkinson-disease-detection.git
cd parkinson-disease-detection
pip install -r requirements.txt
jupyter notebook Parkinson_Detection_Kaggle_Colab.ipynb
```

Set env vars before running:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## 🧬 UI Preview

The Gradio interface features:
- **Full-screen 3D background**: Animated DNA helices (CSS `rotateY` 3D transform) + live synaptic neuron network SVG
- **Frosted-glass panels**: `backdrop-filter: blur(10px)` metric cards
- **Tab 1 — Synaptic Screening**: Upload drawing → instant PD probability + live Accuracy/Precision/Recall/F1 cards
- **Tab 2 — Cohort Benchmark**: Model comparison table with 3-decimal metrics
- **Threshold control**: Default 0.50 or Youden's J optimal threshold toggle

---

## 📁 Repository Structure

```
parkinson-disease-detection/
├── Parkinson_Detection_Kaggle_Colab.ipynb   # Main notebook (18 cells)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt). Key packages:

```
tensorflow >= 2.14
gradio
kagglehub
scikit-learn
scipy
```

---

## ⚠️ Medical Disclaimer

> This project is for **educational and research purposes only**. It is **not** a certified medical device and should **not** be used for clinical diagnosis. Always consult a qualified medical professional for health decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: [Kevin Mader](https://www.kaggle.com/kmader) on Kaggle
- Frameworks: TensorFlow/Keras · Gradio · scikit-learn
- Grad-CAM implementation: Keras-compatible split-pass approach
