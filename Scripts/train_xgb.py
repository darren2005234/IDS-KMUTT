"""
train_xgb.py — XGBoost + Optuna Training Script for KMUTT HPC Cluster
Project  : IDS-KMUTT — Hybrid Intrusion Detection System
Author   : Darren Touopi
Dataset  : CICIDS2017 post-SMOTE (cicids2017_cleaned.csv)

Usage (via SLURM):
    sbatch xgb_job.sbatch

Usage (manual test):
    python train_xgb.py
"""

import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH    = os.path.expanduser("~/ids_project/data/processed/cicids2017_cleaned.csv")
MODEL_DIR    = os.path.expanduser("~/ids_project/models/")
LOG_DIR      = os.path.expanduser("~/ids_project/logs/")
N_TRIALS     = 50
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Tee — write to stdout AND log file simultaneously
log_path = os.path.join(LOG_DIR, "xgb_training.log")
class Tee:
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
print("  IDS-KMUTT — XGBoost + Optuna Training")
print("=" * 60)
print(f"\n[1/6] Loading dataset: {DATA_PATH}")

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
# 2. SPLIT
# ──────────────────────────────────────────────
print("\n[2/6] Stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"      Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

# ──────────────────────────────────────────────
# 3. OPTUNA OBJECTIVE
# ──────────────────────────────────────────────
def objective(trial):
    params = {
        'objective'        : 'binary:logistic',
        'eval_metric'      : 'logloss',
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'use_label_encoder': False,
        'n_jobs'           : -1,
        'random_state'     : RANDOM_STATE
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

# ──────────────────────────────────────────────
# 4. OPTUNA STUDY
# ──────────────────────────────────────────────
print(f"\n[3/6] Optuna study — {N_TRIALS} trials (TPE Bayesian optimisation)...")
print(f"      n_jobs=-1 — using all {os.cpu_count()} available CPU cores")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
t_optuna = time.time() - t0

print(f"\n      Optuna done in {t_optuna/60:.1f} min")
print(f"      Best trial F1 : {study.best_value:.4f}")
print(f"      Best params   : {study.best_params}")

# ──────────────────────────────────────────────
# 5. TRAIN FINAL MODEL
# ──────────────────────────────────────────────
print("\n[4/6] Training final model with best params...")
best_params = study.best_params
best_params.update({
    'objective'        : 'binary:logistic',
    'eval_metric'      : 'logloss',
    'use_label_encoder': False,
    'n_jobs'           : -1,
    'random_state'     : RANDOM_STATE
})

t0 = time.time()
xgb_best = xgb.XGBClassifier(**best_params)
xgb_best.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
t_train = time.time() - t0
print(f"      Final training done in {t_train:.1f}s")

# ──────────────────────────────────────────────
# 6. EVALUATE
# ──────────────────────────────────────────────
print("\n[5/6] Evaluating on test set...")
t0 = time.time()
y_pred = xgb_best.predict(X_test)
y_prob = xgb_best.predict_proba(X_test)[:, 1]
t_infer = time.time() - t0

acc  = accuracy_score(y_test, y_pred)
f1m  = f1_score(y_test, y_pred, average='macro')
f1b  = f1_score(y_test, y_pred, average='binary')
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn)

# Threshold tuning — find best threshold (recall >= 0.95, min FPR)
thresholds_range = np.arange(0.1, 0.95, 0.05)
best_thresh = 0.5
best_fpr_val = 1.0
for thresh in thresholds_range:
    y_pred_t = (y_prob >= thresh).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    fpr_t = fp_t / (fp_t + tn_t)
    if rec_t >= 0.95 and fpr_t < best_fpr_val:
        best_fpr_val = fpr_t
        best_thresh = round(thresh, 2)

print("\n" + "=" * 60)
print("  RESULTS — XGBoost")
print("=" * 60)
print(f"  Accuracy         : {acc:.4f}")
print(f"  F1 (macro)       : {f1m:.4f}")
print(f"  F1 (binary)      : {f1b:.4f}")
print(f"  Precision        : {prec:.4f}")
print(f"  Recall (TPR)     : {rec:.4f}")
print(f"  ROC-AUC          : {auc:.4f}")
print(f"  FPR              : {fpr:.4f}")
print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
print(f"  Best threshold   : {best_thresh} (recall>=0.95, min FPR)")
print(f"  Inference time   : {t_infer:.2f}s ({len(X_test):,} samples)")
print("=" * 60)
print()
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

# ──────────────────────────────────────────────
# 7. SAVE
# ──────────────────────────────────────────────
print("[6/6] Saving model and metrics...")
t_total = time.time() - t0_total

model_path = os.path.join(MODEL_DIR, "xgb_binary_best.joblib")
joblib.dump(xgb_best, model_path)
print(f"      Model saved  : {model_path}")

metrics = {
    "model"         : "XGBoost",
    "accuracy"      : round(acc, 4),
    "f1_macro"      : round(f1m, 4),
    "f1_binary"     : round(f1b, 4),
    "precision"     : round(prec, 4),
    "recall"        : round(rec, 4),
    "roc_auc"       : round(auc, 4),
    "fpr"           : round(fpr, 4),
    "tp"            : int(tp),
    "fp"            : int(fp),
    "tn"            : int(tn),
    "fn"            : int(fn),
    "best_threshold": best_thresh,
    "n_trials"      : N_TRIALS,
    "best_cv_f1"    : round(study.best_value, 4),
    "optuna_time_s" : round(t_optuna, 1),
    "train_time_s"  : round(t_train, 1),
    "infer_time_s"  : round(t_infer, 2),
    "total_time_s"  : round(t_total, 1),
    "best_params"   : str(study.best_params)
}
metrics_path = os.path.join(MODEL_DIR, "xgb_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"      Metrics saved: {metrics_path}")

fi_df = pd.DataFrame({"feature": feature_names, "importance": xgb_best.feature_importances_})\
          .sort_values("importance", ascending=False).reset_index(drop=True)
fi_path = os.path.join(MODEL_DIR, "xgb_feature_importances.csv")
fi_df.to_csv(fi_path, index=False)
print(f"      Top features : {fi_path}")

print(f"\n✅ Total wall time : {t_total/60:.1f} min")
print(f"   Log saved to   : {log_path}")

log_file.close()
