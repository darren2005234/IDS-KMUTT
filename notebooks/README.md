# IDS-KMUTT — Hybrid Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Snort](https://img.shields.io/badge/Snort-2.9.20-red)
![NFStream](https://img.shields.io/badge/NFStream-6.6.0-teal)
![Django](https://img.shields.io/badge/Django-4.x-green?logo=django)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A real-time hybrid intrusion detection system combining ML cascade detection with Snort signature rules.**

*Research internship — KMUTT Bangkok, April–August 2026*  
*Darren Touopi — Polytech Dijon SQR 4A*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [References](#references)

---

## Overview

IDS-KMUTT is a **hybrid intrusion detection system** built as a research project at KMUTT Bangkok. It combines:

- **Three ML models** (Random Forest, XGBoost, LSTM) in a cascade voting architecture
- **Snort 2.9.20** with Emerging Threats Open ruleset (133,866 active rules)
- **NFStream** for real-time bidirectional flow feature extraction
- **Django** dashboard for live monitoring and alert management

The system detects 8 attack classes from CICIDS2017: DDoS, PortScan, DoS, FTP-Patator, SSH-Patator, Botnet, Web Attack, and Heartbleed.

---

## Key Findings

### v2 — Feature Gap Elimination

The main scientific contribution of this project is the identification and elimination of the **feature gap** between training and production:

| Attack | v1 (CICFlowMeter) | v2 (NFStream) | Δ |
|---|---|---|---|
| PortScan | 46.91% | **100.00%** | +53 pp |
| Botnet | ~37.6% | **99.68%** | +62 pp |
| Heartbleed | 61.11% | **100.00%** | +39 pp |
| DoS | 92.97% | **99.99%** | +7 pp |
| Web Attack | 95.78% | **99.86%** | +4 pp |
| DDoS | 99.15% | **100.00%** | +1 pp |

**Root cause**: in v1, CICFlowMeter was used for training (offline) and production (subprocess), but with different internal parameters, creating a feature distribution mismatch. In v2, NFStream is used for both — same code, same parameters, zero gap.

### Live Demo Highlights

- **91.2% BOTH** agreement (ML + Snort confirming simultaneously)
- **DoS is ML_ONLY**: application-layer DoS (slowloris, GoldenEye, Hulk) is invisible to Snort at packet level — ML detected 8,053+ flows Snort missed entirely
- **SSH-Patator**: LSTM (97%) + custom Snort rule SID 9000001 = perfect corroboration
- Detection latency: **< 1s** first flow captured (vs ~30s in v1)

---

## Architecture

```
Network Traffic (eth0)
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  NFStream         │     │  Snort 2.9.20         │
│  Streaming        │     │  133,866 rules        │
│  61 features      │     │  Emerging Threats     │
└────────┬──────────┘     └──────────┬───────────┘
         │                           │
         ▼                           │
┌───────────────────────────────┐    │
│  ML Cascade                   │    │
│  ┌─────┐ ┌────────┐ ┌──────┐ │    │
│  │ RF  │ │XGBoost │ │ LSTM │ │    │
│  └──┬──┘ └───┬────┘ └──┬───┘ │    │
│     └────────┴──────────┘     │    │
│         Voting ≥ 1            │    │
│              │                │    │
│              ▼                │    │
│     Multiclass XGBoost        │    │
│     (attack type)             │    │
└──────────────┬────────────────┘    │
               │                     │
               ▼                     ▼
         ┌─────────────────────────────┐
         │       Fusion Engine         │
         │  R1: ATTACK + Snort → THREAT│
         │  R2: ATTACK only  → ALERT   │
         │  R3: Snort only   → THREAT  │
         │  R4: Neither      → BENIGN  │
         └──────────────┬──────────────┘
                        │
                        ▼
              Django Dashboard
              (real-time monitoring)
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

| Property | Value |
|---|---|
| Total flows (NFStream v2) | 1,845,604 |
| Benign flows | 1,582,456 |
| Attack flows | 263,148 |
| Features (NFStream) | 61 bidirectional |
| Days | Monday–Friday |

| Attack Class | Flows |
|---|---|
| PortScan | 158,720 |
| DDoS | 63,087 |
| DoS (Hulk/GoldenEye/slowloris/Slowhttptest) | 30,365 |
| FTP-Patator | 3,973 |
| SSH-Patator | 2,980 |
| Web Attack (BruteForce/XSS/SQLi) | 2,113 |
| Botnet | 1,902 |
| Heartbleed | 8 |

> **Labelling**: flows are labelled by source/destination IP + timestamp bounds per day, using `scripts/label_nfstream.py`.

---

## Models

All models are trained on NFStream features with hyperparameter optimization.

| Model | Task | F1 macro | Optimization | Notes |
|---|---|---|---|---|
| Random Forest | Binary | 99.96% | GridSearch (48 combos) | max_features=sqrt, n_estimators=200 |
| Random Forest | Multiclass | 95.45% | GridSearch | 9 classes |
| XGBoost | Binary | 99.98% | Optuna TPE (50 trials) | lr=0.131, depth=9, n=248 |
| XGBoost | Multiclass | 99.45% | Optuna TPE (50 trials) | lr=0.039, depth=7, n=469 |
| LSTM | Binary | 99.61% | Optuna (30 trials) | Epoch 13, EarlyStopping |
| LSTM | Multiclass | 77.54% | — | Web Attack 7% — XGBoost covers |

**Cascade decision**: voting ≥ 1 model = ATTACK. XGBoost is the priority multiclass classifier.

**Model files** are tracked with Git LFS (`models/*.joblib`, `models/*.keras`).

---

## Results

### Offline Evaluation (evaluate_v2.py on cicids2017_nfstream_labeled.csv)

| Attack | N flows | Recall | F1 | 3/3 votes | Typed OK |
|---|---|---|---|---|---|
| DDoS | 63,087 | **100.00%** | 99.64% | 99.7% | 100.0% |
| PortScan | 158,720 | **100.00%** | 99.86% | 99.5% | 100.0% |
| FTP-Patator | 3,973 | **100.00%** | 94.54% | 99.4% | 100.0% |
| SSH-Patator | 2,980 | 99.97% | 92.83% | 98.4% | 100.0% |
| DoS | 30,365 | 99.99% | 99.25% | 88.4% | 100.0% |
| Botnet | 1,902 | 99.68% | 89.08% | 57.6% | 99.4% |
| Web Attack | 2,113 | 99.86% | 90.13% | 4.4% | 99.6% |
| Heartbleed | 8 | **100.00%** | 3.37%* | 37.5% | 100.0% |

*\* F1 low due to extreme imbalance (8 flows vs 1.58M benign). Recall confirms correct detection.*

### Live Demo (tcpreplay + real WSL attacks)

| Attack | RF | XGBoost | LSTM | Snort SIDs | BOTH | Type OK |
|---|---|---|---|---|---|---|
| PortScan | 95% | 95% | 95% | 1418, 1421 | 109,684 | ✅ 99.9% |
| SSH-Patator | 53% | 55% | **97%** | 9000001 | 2,886 | ✅ 99.9% |
| FTP-Patator | 11% | **96%** | **96%** | 2417, 491 | 2,121 | ❌ Unknown* |
| DDoS | 32% | **71%** | 18% | 2925 | 0** | ✅ 99.7% |
| DoS | 7% | **12%** | 0.3% | ❌ none | 0 | ✅ 63% |
| Botnet | 7% | 7% | 6% | 2925*** | 0 | ⚠️ 66% |
| Web Attack | 42% | **64%** | 9% | ❌ none | 0 | ❌ 4.5% |
| Heartbleed | 0.9% | 2.6% | 0% | ❌ SID 30514 missing | 0 | ❌ |

*\* FTP-Patator multiclass fails — RST saturation. Snort SID 2417 provides correct type.*  
*\*\* 0 BOTH on DDoS: Snort operates at packet level, NFStream at flow level — different granularities.*  
*\*\*\* Botnet biased by tcpreplay (incomplete C&C sessions). Offline recall: 99.68%.*

---

## Project Structure

```
IDS-KMUTT/
├── dashboard/                    # Django app
│   ├── cicflow_realtime_v1.py    # v1 pipeline (CICFlowMeter — legacy)
│   ├── nfstream_realtime.py      # v2 pipeline (NFStream streaming — production)
│   ├── ml_engine.py              # ML inference (v1/v2 autodetect)
│   ├── fusion_engine.py          # Decision rules (R1-R4)
│   ├── models.py                 # Alert Django model
│   ├── views.py                  # API + dashboard views
│   ├── templates/dashboard/      # HTML templates
│   └── static/dashboard/         # CSS + JS
│
├── models/                       # Trained models (Git LFS)
│   ├── rf_binary_v2.joblib
│   ├── rf_multiclass_v2.joblib
│   ├── xgb_binary_v2.joblib
│   ├── xgb_multiclass_v2.joblib
│   ├── lstm_binary_v2.keras
│   ├── lstm_multiclass_v2.keras
│   └── scaler_nfstream.joblib
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_exploration.ipynb      # Dataset exploration
│   ├── 02_preprocessing.ipynb    # NFStream extraction + labelling
│   ├── 03_random_forest.ipynb    # RF training (GridSearch)
│   ├── 04_xgboost.ipynb          # XGBoost training (Optuna)
│   ├── 05_lstm.ipynb             # LSTM training (Optuna)
│   └── 06_benchmark.ipynb        # Full benchmark + feature gap analysis
│
├── scripts/                      # Cluster scripts (KMUTT HPC)
│   ├── extract_nfstream.py       # NFStream PCAP extraction
│   ├── label_nfstream.py         # IP+timestamp labelling
│   ├── preprocess_nfstream.py    # Scaling + SMOTE + split
│   ├── train_rf_v2.py            # RF training
│   ├── train_xgb_v2.py           # XGBoost + Optuna
│   └── train_lstm_v2.py          # LSTM + Optuna
│
├── evaluate_v2.py                # v2 evaluation script (NFStream CSV)
├── evaluate.py                   # v1 evaluation script (CICFlowMeter CSV)
├── feature_names_nfstream.txt    # 61 NFStream feature names
├── manage.py                     # Django management
├── ids_kmutt/settings.py         # Django settings
└── requirements.txt              # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.12+
- Snort 2.9.20
- NFStream 6.6.0
- Ubuntu 22.04+ (VM or bare metal)

### Setup

```bash
# Clone repository
git clone https://github.com/darren2005234/IDS-KMUTT.git
cd IDS-KMUTT

# Install Git LFS (for model files)
git lfs install
git lfs pull

# Create virtual environment
python3 -m venv ids_env
source ids_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Django database
python manage.py migrate

# Load models and start
sudo systemctl start ids-snort
sudo systemctl start ids-django
sudo systemctl start ids-pipeline
```

### Requirements

```
nfstream==6.6.0
tensorflow>=2.15
scikit-learn>=1.4
xgboost>=2.0
optuna>=3.5
imbalanced-learn>=0.12
django>=4.2
djangorestframework>=3.14
pandas>=2.0
numpy>=1.26
joblib>=1.3
requests>=2.31
matplotlib>=3.8
seaborn>=0.13
```

---

## Usage

### Start the live pipeline

```bash
# Activate environment
source ids_env/bin/activate

# Start all services
sudo systemctl start ids-snort ids-django ids-pipeline

# Monitor logs
sudo journalctl -u ids-pipeline -f
```

### Access the dashboard

```
http://<VM_IP>:8000/
```

### Run offline evaluation (v2)

```bash
# Evaluate all attack classes on NFStream CSV
python evaluate_v2.py --day all --compare-modes

# Evaluate a specific day
python evaluate_v2.py --day friday --compare-modes \
  --csv ~/data/cicids2017_nfstream_labeled.csv
```

### Train models on HPC cluster (KMUTT CPE Slurm)

```bash
# Submit extraction job (anchor to gpu02 — NFS mounted)
sbatch --nodelist=gpu02 scripts/extract_nfstream.sbatch

# Submit training jobs
sbatch scripts/train_rf_v2.sbatch
sbatch scripts/train_xgb_v2.sbatch
sbatch scripts/train_lstm_v2.sbatch
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_exploration.ipynb` | CICIDS2017 dataset exploration, class distribution, feature correlation |
| `02_preprocessing.ipynb` | NFStream extraction, IP-based labelling, SMOTE, StandardScaler |
| `03_random_forest.ipynb` | RF training with GridSearchCV, feature importance |
| `04_xgboost.ipynb` | XGBoost training with Optuna TPE (50 trials) |
| `05_lstm.ipynb` | LSTM training with sliding window sequences + Optuna |
| `06_benchmark.ipynb` | **Full benchmark**: binary metrics, multiclass heatmap, cascade analysis, feature gap visualization |

---

## Known Limitations

1. **LSTM Multiclass on Web Attack**: 7% recall — Web Attack has no exploitable temporal pattern. XGBoost multiclass covers this class.
2. **Heartbleed**: Only 8 flows in CICIDS2017. SID 30514 (Talos) missing from Emerging Threats Open. Add manually to `local.rules`.
3. **FTP-Patator multiclass typing**: RST saturation causes classifier failure. Snort SID 2417 compensates.
4. **Botnet via tcpreplay**: Incomplete C&C sessions (no real TCP handshake) degrade NFStream features. Offline recall: 99.68%.
5. **Concept drift**: Models trained on 2017 synthetic benign traffic generate false positives on modern system traffic (Ubuntu apt, GitHub, Azure metadata service).

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization.* ICISSP.
2. Ring, M., et al. (2019). *A survey of network-based intrusion detection data sets.* Computers & Security.
3. Ferrag, M. A., et al. (2020). *Deep learning for cyber security intrusion detection.* Journal of Information Security and Applications.
4. Breiman, L. (2001). *Random forests.* Machine Learning.
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
6. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation.

---

## Citation

```bibtex
@misc{touopi2026ids,
  title   = {IDS-KMUTT: A Hybrid Real-Time Intrusion Detection System 
             with NFStream Feature Gap Elimination},
  author  = {Touopi, Darren},
  year    = {2026},
  school  = {Polytech Dijon / KMUTT Bangkok},
  note    = {Research internship project, target: ARES/ICISSP}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with ❤️ at KMUTT Bangkok · Polytech Dijon SQR 4A · 2026</sub>
</div>
