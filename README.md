# LENS-XAI 🔍
### Explainable AI for Network Intrusion Detection

A deep learning pipeline that detects network intrusions **and explains why** - combining VAE-based representation learning, knowledge distillation, and LLM-powered natural language explanations.

---

## 📌 Overview

Network intrusion detection systems are often black boxes - they flag traffic as malicious but don't explain what triggered the alert. **LENS-XAI** addresses this by pairing a high-accuracy detection model with an explainability layer that tells security analysts *which network features* drove each prediction, in plain English.

---

## 🏗️ Architecture

```
Raw Network Traffic (CICIDS2017)
        ↓
  Preprocessing + SMOTE
        ↓
  VAE (Encoder) → Latent Space
        ↓
  Teacher Model → Knowledge Distillation → Student Model
        ↓
  Shapley-style Attribution (XAI)
        ↓
  LLM (Groq / Gemini) → Natural Language Explanation
```

---

## ⚙️ Features

- **VAE-based feature extraction** -learns compact latent representations of network traffic
- **Teacher-Student distillation** - compresses a large teacher model (512→256→128) into a lightweight student model (64→32) with minimal accuracy loss
- **Shapley-style XAI** - ranks top contributing features (e.g. packet length, flow duration) per prediction
- **LLM integration** - auto-generates human-readable explanations via Groq (Llama 4), Gemini Flash, or Ollama
- **Class imbalance handling** - SMOTE oversampling + weighted CrossEntropyLoss
- **Multi-class detection** - classifies 10+ attack categories (DoS, DDoS, PortScan, Brute Force, etc.)

---

## 📊 Dataset

**CICIDS2017** - Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

Download it from: https://www.unb.ca/cic/datasets/ids-2017.html

> ⚠️ The dataset is not included in this repo due to its size. Place the CSV files in a `/data` folder after downloading.

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

1. Open the notebook directly in Colab

2. Download the CICIDS2017 dataset and upload the CSV files to your **Google Drive**

3. Mount your Drive in the notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Update the dataset path in the notebook to point to your Drive folder:
```python
DATA_PATH = "/content/drive/MyDrive/CICIDS2017/"
```

5. Run all cells - install commands (`!pip install ...`) are already included in the notebook

---

## 🔑 LLM API Setup

The project supports multiple free LLM providers. Set your preferred API key as an environment variable:

```bash
# Groq (Llama 4)
export GROQ_API_KEY=your_key_here

# OR Gemini Flash
export GEMINI_API_KEY=your_key_here

# OR run Ollama locally (no key needed)
```

---

## 📦 Dependencies

All dependencies are installed directly within the Colab notebook using `!pip install`. No separate `requirements.txt` needed.

---

## 📈 Results

| Metric | Teacher Model | Student Model |
|--------|--------------|---------------|
| Accuracy | 73.81% | 75.63% |
| Precision | 93.39% | 93.55% |
| Recall | 73.81% | 75.63% |
| F1-Score | 80.23% | 81.56% |
| Parameters | 200,844 | 17,740 |
| Inference Time | 3.42 ms/batch | 2.98 ms/batch |

> 🏆 The Student model outperforms the Teacher on all metrics while using **91.2% fewer parameters** and running **13% faster** - demonstrating effective knowledge distillation.

**Attack Classes Detected:**
`Benign`, `Bot`, `DDoS`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowhttptest`, `DoS-Slowloris`, `FTP-Patator`, `PortScan`, `SSH-Patator`, `Web Attack - Brute Force`, `Web Attack - XSS`

### 🔍 XAI Attribution - Feature Contributions

![XAI Attribution with LLM Explanations]<img width="2443" height="1485" alt="cicids2017_attributions_llm" src="https://github.com/user-attachments/assets/9d597fe7-eb1a-4ed1-aa45-c5542ff824ab" />


The plot shows Shapley-style feature attributions for both Teacher and Student models on a sample prediction (classified as **Bot** traffic):

- **Teacher Model** - Top positive contributors: `URG Flag Count`, `Down/Up Ratio`, `Bwd Packets/s`. Top negative features: `Bwd Packet Length Min`, `Fwd Packet Length Min`
- **Student Model** - Top positive contributors: `URG Flag Count`, `Down/Up Ratio`, `Min Packet Length`. Strongest suppressor: `ACK Flag Count` (~−0.06)
- Both models agree on core influential features, validating that the distilled student preserves the teacher's decision logic

### 🤖 LLM Explanation Sample (Bot Traffic)

**Teacher:** Flagged `URG Flag Count`, `Down/Up Ratio`, and `Packet Length Mean` as indicators of abnormal Bot-like traffic. Normal characteristics like `Min Packet Length` and `PSH Flag Count` reduced the probability. Prediction advised with caution given moderate accuracy.

**Student:** Agreed on `URG Flag Count` and `Down/Up Ratio` as key Bot indicators. Diverged on `ACK Flag Count` and `Init_Win_bytes_backward` as suppressing features - slightly different reasoning path but consistent final conclusion.

**Comparison:** Both models converge on the same prediction with overlapping top features, though minor discrepancies in feature attribution directions highlight the inherent trade-off in knowledge distillation.

---

## 🗂️ Project Structure

```
lens-xai/
├── lens_xai_colab_final.py   # Main Colab notebook/script
├── results/                   # Output plots and reports (generated at runtime)
└── README.md
```

> 📁 Dataset is stored separately on Google Drive and not included in the repo.

---
