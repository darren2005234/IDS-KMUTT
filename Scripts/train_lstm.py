"""
train_lstm.py — LSTM Training Script for KMUTT HPC Cluster
Project  : IDS-KMUTT — Hybrid Intrusion Detection System
Author   : Darren Touopi
Dataset  : CICIDS2017 post-SMOTE (cicids2017_cleaned.csv)
Reference: Hochreiter & Schmidhuber (1997), Ferrag et al. (2020)

Sequence length: 10 consecutive flows (sliding window)
GPU: NVIDIA RTX 4090 (24GB VRAM) — gpu4090 partition

Usage (via SLURM):
    sbatch lstm_job.sbatch

Usage (manual test):
    python train_lstm.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# TensorFlow config — GPU memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH    = os.path.expanduser("~/data/cicids2017_cleaned.csv")
MODEL_DIR    = os.path.expanduser("~/ids_project/models/")
LOG_DIR      = os.path.expanduser("~/ids_project/logs/")
SEQUENCE_LEN = 10        # Ferrag et al. (2020) — 10 consecutive flows
BATCH_SIZE   = 512       # Large batch for RTX 4090
EPOCHS       = 30
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Tee — write to stdout AND log file
log_path = os.path.join(LOG_DIR, "lstm_training.log")
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
# GPU SETUP
# ──────────────────────────────────────────────
print("=" * 60)
print("  IDS-KMUTT — LSTM Training")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"\nGPU detected  : {len(gpus)} × {gpus[0].name}")
else:
    print("\nNo GPU detected — running on CPU")

print(f"TensorFlow    : {tf.__version__}")
print(f"Sequence len  : {SEQUENCE_LEN}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Max epochs    : {EPOCHS}")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print(f"\n[1/7] Loading dataset: {DATA_PATH}")
t0_total = time.time()

df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

LABEL_COLS = ["Label_binary"]
X = df.drop(columns=LABEL_COLS).values.astype(np.float32)
y = df["Label_binary"].values.astype(np.float32)

print(f"      Features  : {X.shape[1]}")
print(f"      Class dist: BENIGN={int((y==0).sum()):,} | ATTACK={int((y==1).sum()):,}")

# ──────────────────────────────────────────────
# 2. NORMALIZE
# ──────────────────────────────────────────────
print("\n[2/7] Normalizing features (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for inference
import joblib
scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"      Scaler saved: {scaler_path}")

# ──────────────────────────────────────────────
# 3. CREATE SEQUENCES (sliding window)
# ──────────────────────────────────────────────
print(f"\n[3/7] Creating sequences (window={SEQUENCE_LEN})...")
print(f"      This may take a few minutes for 4.5M rows...")

t0 = time.time()

# Subsample to 1M for sequence creation — memory constraint
# Full dataset sequences would require ~100GB RAM
SAMPLE_SIZE = 1_000_000
if len(X_scaled) > SAMPLE_SIZE:
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_scaled), SAMPLE_SIZE, replace=False)
    idx = np.sort(idx)  # Keep temporal order
    X_seq_data = X_scaled[idx]
    y_seq_data = y[idx]
    print(f"      Subsampled to {SAMPLE_SIZE:,} rows for sequence creation")
else:
    X_seq_data = X_scaled
    y_seq_data = y

# Create sequences using stride 1
n_sequences = len(X_seq_data) - SEQUENCE_LEN
X_sequences = np.zeros((n_sequences, SEQUENCE_LEN, X_seq_data.shape[1]), dtype=np.float32)
y_sequences = np.zeros(n_sequences, dtype=np.float32)

for i in range(n_sequences):
    X_sequences[i] = X_seq_data[i:i + SEQUENCE_LEN]
    y_sequences[i] = y_seq_data[i + SEQUENCE_LEN - 1]

t_seq = time.time() - t0
print(f"      Sequences created: {X_sequences.shape} in {t_seq:.1f}s")
print(f"      Memory usage     : {X_sequences.nbytes / 1e9:.2f} GB")

# ──────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[4/7] Stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_sequences
)
print(f"      Train : {X_train.shape[0]:,} sequences")
print(f"      Test  : {X_test.shape[0]:,} sequences")

# ──────────────────────────────────────────────
# 5. BUILD LSTM MODEL
# ──────────────────────────────────────────────
print("\n[5/7] Building LSTM model...")

model = Sequential([
    # First LSTM layer — captures short-term temporal patterns
    LSTM(128, input_shape=(SEQUENCE_LEN, X.shape[1]),
         return_sequences=True, name='lstm_1'),
    BatchNormalization(),
    Dropout(0.3),

    # Second LSTM layer — captures longer-term dependencies
    LSTM(64, return_sequences=False, name='lstm_2'),
    BatchNormalization(),
    Dropout(0.3),

    # Dense layers — classification head
    Dense(32, activation='relu', name='dense_1'),
    Dropout(0.2),
    Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

total_params = model.count_params()
print(f"\n      Total parameters: {total_params:,}")

# ──────────────────────────────────────────────
# 6. TRAIN
# ──────────────────────────────────────────────
print(f"\n[6/7] Training LSTM — {EPOCHS} epochs max...")

checkpoint_path = os.path.join(MODEL_DIR, "lstm_checkpoint.keras")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
t_train = time.time() - t0

print(f"\n      Training completed in {t_train/60:.1f} min")
print(f"      Epochs run: {len(history.history['loss'])}")

# ──────────────────────────────────────────────
# 7. EVALUATE
# ──────────────────────────────────────────────
print("\n[7/7] Evaluating on test set...")

t0 = time.time()
y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
t_infer = time.time() - t0

acc  = accuracy_score(y_test, y_pred)
f1m  = f1_score(y_test, y_pred, average='macro')
f1b  = f1_score(y_test, y_pred, average='binary')
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn)

# Threshold tuning
best_thresh = 0.5
best_fpr_val = 1.0
for thresh in np.arange(0.1, 0.95, 0.05):
    y_pred_t = (y_prob >= thresh).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    fpr_t = fp_t / (fp_t + tn_t)
    if rec_t >= 0.95 and fpr_t < best_fpr_val:
        best_fpr_val = fpr_t
        best_thresh  = round(thresh, 2)

t_total = time.time() - t0_total

print("\n" + "=" * 60)
print("  RESULTS — LSTM")
print("=" * 60)
print(f"  Accuracy         : {acc:.4f}")
print(f"  F1 (macro)       : {f1m:.4f}")
print(f"  F1 (binary)      : {f1b:.4f}")
print(f"  Precision        : {prec:.4f}")
print(f"  Recall (TPR)     : {rec:.4f}")
print(f"  ROC-AUC          : {auc:.4f}")
print(f"  FPR              : {fpr:.4f}")
print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
print(f"  Best threshold   : {best_thresh}")
print(f"  Inference time   : {t_infer:.2f}s ({len(X_test):,} sequences)")
print(f"  Training time    : {t_train/60:.1f} min")
print(f"  Total wall time  : {t_total/60:.1f} min")
print("=" * 60)
print()
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

# Save model
model_path = os.path.join(MODEL_DIR, "lstm_binary_best.keras")
model.save(model_path)
print(f"Model saved      : {model_path}")

# Save metrics
metrics = {
    "model"         : "LSTM",
    "sequence_len"  : SEQUENCE_LEN,
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
    "train_time_s"  : round(t_train, 1),
    "infer_time_s"  : round(t_infer, 2),
    "total_time_s"  : round(t_total, 1),
    "epochs_run"    : len(history.history['loss']),
    "batch_size"    : BATCH_SIZE,
}
metrics_path = os.path.join(MODEL_DIR, "lstm_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"Metrics saved    : {metrics_path}")

print(f"\n✅ LSTM training complete — Total: {t_total/60:.1f} min")
print(f"   Log saved to : {log_path}")

log_file.close()
