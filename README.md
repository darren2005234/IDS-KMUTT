# 🛡️ IDS-KMUTT — Hybrid Network Intrusion Detection System

**Author:** Darren Touopi  
**Institution:** KMUTT — Department of Computer Engineering, Bangkok, Thailand  
**Programme:** BAC+4 SQR — Polytech Dijon (ESIREM), France  
**Internship Duration:** April 30 – August 15, 2026  

---

## 📌 Project Overview

IDS-KMUTT is a hybrid network intrusion detection system designed to protect university campus networks. It combines two complementary detection paradigms:

- 🤖 **Machine Learning** (Random Forest, XGBoost, LSTM) — anomaly detection
- 🔍 **Snort 3** — signature-based detection (Emerging Threats ruleset)
- ⚙️ **Fusion Engine** — combines both layers using 4 decision rules
- 📊 **Real-time Django dashboard** — Wazuh-inspired interface

**Dataset:** CICIDS2017 — 4,542,640 rows after SMOTE balancing  
**Reference:** Sharafaldin et al. (2018), Ring et al. (2019)

---

## 📊 Results

### ✅ Random Forest (Notebook 03)
| Metric | Value | SRS Target |
|---|---|---|
| Accuracy | **99.94%** | — |
| F1 (macro) | **99.94%** | ≥ 97% ✅ |
| Precision | **99.91%** | — |
| Recall (TPR) | **99.96%** | ≥ 95% ✅ |
| ROC-AUC | **1.0000** 🔥 | — |
| FPR | **0.09%** | ≤ 2% ✅ |

**Best params:** `max_depth=30, max_features=sqrt, min_samples_split=5, n_estimators=100`  
**Training time:** 220 min — KMUTT HPC (gpu3080, 16 CPUs, 58 GB RAM)  

### ✅ XGBoost (Notebook 04)
| Metric | Value | SRS Target |
|---|---|---|
| Accuracy | **99.94%** | — |
| F1 (macro) | **99.94%** | ≥ 97% ✅ |
| Precision | **99.91%** | — |
| Recall (TPR) | **99.98%** | ≥ 95% ✅ |
| ROC-AUC | **1.0000** 🔥 | — |
| FPR | **0.09%** | ≤ 2% ✅ |

**Best params:** Optuna TPE — 50 trials  
**Training time:** 84.6 min — KMUTT HPC (gpu3080, 16 CPUs, 58 GB RAM)  
**Inference time:** 1.00s / 908k samples — 2x faster than RF ✅

### 📋 LSTM (Notebook 05)
*Planned — gpu4090 partition (RTX 4090)*

### 📋 Benchmark (Notebook 06)
*Planned — RF vs XGBoost vs LSTM vs Snort vs Hybrid*

---

## 🗂️ Repository Structure

```
IDS-KMUTT/
├── notebooks/              ← Jupyter notebooks (01 → 07)
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_random_forest.ipynb
│   └── 04_xgboost.ipynb
├── Scripts/                ← SLURM job scripts + Python training scripts
│   ├── train_rf.py         ← RF training script for HPC cluster
│   ├── rf_job.sbatch       ← SLURM job script for RF
│   ├── train_xgb.py        ← XGBoost + Optuna training script
│   └── xgb_job.sbatch      ← SLURM job script for XGBoost
├── dashboard/              ← Django app
│   ├── models.py           ← Alert database model
│   ├── views.py            ← REST API + dashboard view
│   ├── fusion_engine.py    ← Hybrid detection logic (4 rules)
│   ├── urls.py             ← URL routing
│   ├── templates/          ← HTML dashboard (Wazuh-inspired)
│   └── static/             ← CSS + JavaScript (Chart.js)
├── results/                ← Metrics per model
│   └── rf/
│       ├── rf_metrics.csv              ← F1, AUC, FPR, accuracy...
│       └── rf_feature_importances.csv  ← Top 50 features ranked
├── report/                 ← Literature notes
│   └── literature_notes.md
└── data/                   ← .gitkeep (dataset not versioned — 1.5 GB)
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Dataset | CICIDS2017 — 4,542,640 rows post-SMOTE |
| ML Training | scikit-learn, XGBoost, TensorFlow |
| HPC | KMUTT CPE Cluster — SLURM (gpu3080, gpu4090) |
| Signature IDS | Snort 3 + Emerging Threats ruleset |
| Dashboard | Django 6.0.5, Chart.js, REST API |
| Deployment | KMUTT VM — Ubuntu 24.04 |

---

## 🚀 Fusion Engine — Decision Rules

| ML Result | Snort Alert | Decision | Source Tag |
|---|---|---|---|
| ATTACK | YES | 🔴 THREAT | Both |
| ATTACK | NO (conf > 85%) | 🟠 ALERT | ML Only |
| BENIGN | YES | 🔴 THREAT | Snort Only |
| BENIGN | NO | 🟢 BENIGN | None |

---

## 📅 Timeline

| Milestone | Status | Date |
|---|---|---|
| Dataset exploration | ✅ Done | April 2026 |
| Preprocessing + SMOTE | ✅ Done | April 2026 |
| Random Forest training | ✅ Done | May 2026 |
| XGBoost training | ✅ Done | May 2026 |
| LSTM training | 📋 Planned | June 2026 |
| Benchmark (Notebook 06) | 📋 Planned | June 2026 |
| IDS testing (Notebook 07) | 📋 Planned | July 2026 |
| Django dashboard | ✅ Done | May 2026 |
| Docker container | 📋 Planned | August 2026 |
| Final report | 📋 Planned | August 2026 |

---

## 📚 References

- Sharafaldin, I. et al. (2018). Toward Generating a New Intrusion Detection Dataset. ICISSP.
- Ring, M. et al. (2019). A Survey of Network-based Intrusion Detection Data Sets.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Ferrag, M. A. et al. (2020). Deep Learning for Cyber Security Intrusion Detection.
