"""
train_rf.py — Random Forest Training Script for KMUTT HPC Cluster
Project  : IDS-KMUTT — Hybrid Intrusion Detection System
Author   : Darren Touopi
Dataset  : CICIDS2017 post-SMOTE (cicids2017_cleaned.csv)

Usage (via SLURM):
    sbatch rf_job.sbatch

Usage (manual test):
    python train_rf.py
"""

import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ──────────────────────────────────────────────
# CONFIG — adjust paths if needed
# ──────────────────────────────────────────────
DATA_PATH  = os.path.expanduser("~/data/cicids2017_cleaned.csv")
MODEL_DIR  = os.path.expanduser("~/ids_project/models/")
LOG_DIR    = os.path.expanduser("~/ids_project/logs/")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Redirect stdout to log file as well
log_path = os.path.join(LOG_DIR, "rf_training.log")

class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  IDS-KMUTT — Random Forest Training")
print("=" * 60)
print(f"\n[1/6] Loading dataset from: {DATA_PATH}")

t0_total = time.time()
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

LABEL_COLS = ["Label_binary", "Label_encoded"]
X = df.drop(columns=LABEL_COLS)
y = df["Label_binary"]
feature_names = X.columns.tolist()

print(f"      Features  : {X.shape[1]}")
print(f"      Class dist: {dict(y.value_counts())}")

# ──────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[2/6] Stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"      Train : {X_train.shape[0]:,} rows")
print(f"      Test  : {X_test.shape[0]:,} rows")

# ──────────────────────────────────────────────
# 3. HYPERPARAMETER GRID
# ──────────────────────────────────────────────
# CPU01 has 64 threads → n_jobs=-1 uses all of them
# Full grid is tractable on the cluster in ~15-30 min
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [None, 20, 30],
    "max_features"     : ["log2", "sqrt"],
    "min_samples_split": [2, 5],
}

n_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n[3/6] GridSearchCV — {n_combinations} combos × 3 folds = {n_combinations*3} fits")
print(f"      Scoring: f1_macro | n_jobs=-1 (all CPU cores)")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rf_cv = RandomForestClassifier(
    class_weight="balanced",
    oob_score=False,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

grid_search = GridSearchCV(
    estimator=rf_cv,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

t0 = time.time()
grid_search.fit(X_train, y_train)
t_grid = time.time() - t0

print(f"\n      GridSearchCV done in {t_grid/60:.1f} min")
print(f"      Best params : {grid_search.best_params_}")
print(f"      Best CV F1  : {grid_search.best_score_:.4f}")

# ──────────────────────────────────────────────
# 4. EVALUATE ON TEST SET
# ──────────────────────────────────────────────
print("\n[4/6] Evaluating on test set...")
rf_best = grid_search.best_estimator_

t0 = time.time()
y_pred = rf_best.predict(X_test)
y_prob = rf_best.predict_proba(X_test)[:, 1]
t_infer = time.time() - t0

acc  = accuracy_score(y_test, y_pred)
f1m  = f1_score(y_test, y_pred, average="macro")
f1b  = f1_score(y_test, y_pred, average="binary")
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn)

print("\n" + "=" * 60)
print("  RESULTS — Random Forest")
print("=" * 60)
print(f"  Accuracy         : {acc:.4f}")
print(f"  F1 (macro)       : {f1m:.4f}")
print(f"  F1 (binary)      : {f1b:.4f}")
print(f"  Precision        : {prec:.4f}")
print(f"  Recall (TPR)     : {rec:.4f}")
print(f"  ROC-AUC          : {auc:.4f}")
print(f"  FPR              : {fpr:.4f}")
print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
print(f"  Inference time   : {t_infer:.2f}s ({len(X_test):,} samples)")
print("=" * 60)
print()
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
print("[5/6] Saving feature importances...")
importances = rf_best.feature_importances_
fi_df = pd.DataFrame({
    "feature"   : feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

fi_path = os.path.join(MODEL_DIR, "rf_feature_importances.csv")
fi_df.to_csv(fi_path, index=False)
print(f"      Saved: {fi_path}")
print(f"\n      Top 10 features:")
print(fi_df.head(10).to_string(index=False))

# ──────────────────────────────────────────────
# 6. SAVE MODEL AND METRICS
# ──────────────────────────────────────────────
print("\n[6/6] Saving model and metrics...")
t_total = time.time() - t0_total

model_path = os.path.join(MODEL_DIR, "rf_binary_best.joblib")
joblib.dump(rf_best, model_path)
print(f"      Model saved  : {model_path}")

metrics = {
    "model"         : "Random Forest",
    "accuracy"      : round(acc,  4),
    "f1_macro"      : round(f1m,  4),
    "f1_binary"     : round(f1b,  4),
    "precision"     : round(prec, 4),
    "recall"        : round(rec,  4),
    "roc_auc"       : round(auc,  4),
    "fpr"           : round(fpr,  4),
    "tp"            : int(tp),
    "fp"            : int(fp),
    "tn"            : int(tn),
    "fn"            : int(fn),
    "train_time_s"  : round(t_grid, 1),
    "infer_time_s"  : round(t_infer, 2),
    "total_time_s"  : round(t_total, 1),
    "best_params"   : str(grid_search.best_params_),
    "best_cv_f1"    : round(grid_search.best_score_, 4),
}

metrics_path = os.path.join(MODEL_DIR, "rf_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"      Metrics saved: {metrics_path}")

print(f"\n Total wall time: {t_total/60:.1f} min")
print(f"   Log saved to  : {log_path}")

log_file.close()
