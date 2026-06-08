"""
ml_engine.py — ML Model Loading and Inference
Project : IDS-KMUTT — Hybrid Intrusion Detection System
Author  : Darren Touopi

Cascade detection pipeline:
  1. All 3 binary models run simultaneously (RF, XGBoost, LSTM)
  2. Voting: at least 1 vote = ATTACK
  3. If ATTACK → multiclass model identifies the attack type
"""

import os
import joblib
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Model paths ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RF_BIN_PATH       = os.path.join(MODELS_DIR, 'rf_binary_best.joblib')
XGB_BIN_PATH      = os.path.join(MODELS_DIR, 'xgb_binary_best.joblib')
LSTM_BIN_PATH     = os.path.join(MODELS_DIR, 'lstm_binary_best.keras')
LSTM_BIN_SCALER   = os.path.join(MODELS_DIR, 'lstm_scaler.joblib')

RF_MULTI_PATH     = os.path.join(MODELS_DIR, 'rf_multiclass_best.joblib')
XGB_MULTI_PATH    = os.path.join(MODELS_DIR, 'xgb_multiclass_best.joblib')
LSTM_MULTI_PATH   = os.path.join(MODELS_DIR, 'lstm_multiclass_best.keras')
LSTM_MULTI_SCALER = os.path.join(MODELS_DIR, 'lstm_multi_scaler.joblib')

# ── Class mapping ─────────────────────────────────────────────
MULTICLASS_LABELS = {
    0: 'BENIGN',
    1: 'Botnet',
    2: 'Brute Force',
    3: 'DDoS',
    4: 'DoS',
    5: 'PortScan',
    6: 'Rare Attack',
    7: 'Web Attack',
}

# ── Feature lists ─────────────────────────────────────────────
# Binary models — 50 features (trained on cicids2017_cleaned.csv)
FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Variance',
    'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std'
]

# Multiclass models — 47 features (after feature selection on cicids2017_multiclass.csv)
FEATURE_NAMES_MULTI = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Variance',
    'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Std'
]

SEQUENCE_LEN = 10

# ── Model storage — dict avoids global variable issues ────────
_models = {
    'rf_bin'      : None,
    'xgb_bin'     : None,
    'lstm_bin'    : None,
    'scaler_bin'  : None,
    'rf_multi'    : None,
    'xgb_multi'   : None,
    'lstm_multi'  : None,
    'scaler_multi': None,
}

_seq_buffer_bin   = []
_seq_buffer_multi = []


def load_models():
    """Load all ML models into _models dict."""

    if os.path.exists(RF_BIN_PATH):
        print("[ML Engine] Loading RF binary...")
        _models['rf_bin'] = joblib.load(RF_BIN_PATH)
        print("[ML Engine] RF binary loaded ✅")

    if os.path.exists(XGB_BIN_PATH):
        print("[ML Engine] Loading XGBoost binary...")
        _models['xgb_bin'] = joblib.load(XGB_BIN_PATH)
        print("[ML Engine] XGBoost binary loaded ✅")

    if os.path.exists(LSTM_BIN_PATH) and os.path.exists(LSTM_BIN_SCALER):
        print("[ML Engine] Loading LSTM binary...")
        import tensorflow as tf
        _models['lstm_bin']   = tf.keras.models.load_model(LSTM_BIN_PATH)
        _models['scaler_bin'] = joblib.load(LSTM_BIN_SCALER)
        print("[ML Engine] LSTM binary loaded ✅")

    if os.path.exists(RF_MULTI_PATH):
        print("[ML Engine] Loading RF multiclass...")
        _models['rf_multi'] = joblib.load(RF_MULTI_PATH)
        print("[ML Engine] RF multiclass loaded ✅")

    if os.path.exists(XGB_MULTI_PATH):
        print("[ML Engine] Loading XGBoost multiclass...")
        _models['xgb_multi'] = joblib.load(XGB_MULTI_PATH)
        print("[ML Engine] XGBoost multiclass loaded ✅")

    if os.path.exists(LSTM_MULTI_PATH) and os.path.exists(LSTM_MULTI_SCALER):
        print("[ML Engine] Loading LSTM multiclass...")
        import tensorflow as tf
        _models['lstm_multi']   = tf.keras.models.load_model(LSTM_MULTI_PATH)
        _models['scaler_multi'] = joblib.load(LSTM_MULTI_SCALER)
        print("[ML Engine] LSTM multiclass loaded ✅")


def predict(features: dict, model_name: str = None, mode: str = None) -> dict:
    """
    Cascade detection pipeline:
    1. Run all 3 binary models simultaneously
    2. Vote: at least 1 ATTACK = final ATTACK
    3. If ATTACK → multiclass identifies the attack type

    Args:
        features : dict of feature_name -> value
        model_name, mode : ignored (kept for backward compatibility)

    Returns:
        dict with ml_prediction, ml_confidence, attack_type, votes, etc.
    """

    # Load models if not already loaded
    if _models['rf_bin'] is None and _models['xgb_bin'] is None:
        load_models()

    # Build feature DataFrames
    X_bin   = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURE_NAMES}])
    X_multi = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURE_NAMES_MULTI}])

    results = {}

    # ── Random Forest binary ──────────────────────────────────
    if _models['rf_bin'] is not None:
        try:
            pred = int(_models['rf_bin'].predict(X_bin)[0])
            conf = float(_models['rf_bin'].predict_proba(X_bin)[0][1])
            results['RandomForest'] = {'pred': pred, 'conf': round(conf, 4)}
        except Exception as e:
            print(f"[ML Engine] RF binary error: {e}")
            results['RandomForest'] = {'pred': 0, 'conf': 0.0}

    # ── XGBoost binary ────────────────────────────────────────
    if _models['xgb_bin'] is not None:
        try:
            pred = int(_models['xgb_bin'].predict(X_bin)[0])
            conf = float(_models['xgb_bin'].predict_proba(X_bin)[0][1])
            results['XGBoost'] = {'pred': pred, 'conf': round(conf, 4)}
        except Exception as e:
            print(f"[ML Engine] XGBoost binary error: {e}")
            results['XGBoost'] = {'pred': 0, 'conf': 0.0}

    # ── LSTM binary ───────────────────────────────────────────
    if _models['lstm_bin'] is not None and _models['scaler_bin'] is not None:
        try:
            #X_scaled = _models['scaler_bin'].transform(X_bin.values)[0]
            X_scaled = np.clip(_models['scaler_bin'].transform(X_bin.values), -10, 10)[0]
            _seq_buffer_bin.append(X_scaled)
            if len(_seq_buffer_bin) > SEQUENCE_LEN:
                _seq_buffer_bin.pop(0)
            if len(_seq_buffer_bin) == SEQUENCE_LEN:
                X_seq = np.array(_seq_buffer_bin, dtype=np.float32)
                X_seq = X_seq.reshape(1, SEQUENCE_LEN, len(FEATURE_NAMES))
                conf  = float(_models['lstm_bin'].predict(X_seq, verbose=0)[0][0])
                pred  = 1 if conf >= 0.9 else 0
            else:
                pred, conf = 0, 0.0
            results['LSTM'] = {'pred': pred, 'conf': round(conf, 4)}
        except Exception as e:
            print(f"[ML Engine] LSTM binary error: {e}")
            results['LSTM'] = {'pred': 0, 'conf': 0.0}

    # ── Voting ────────────────────────────────────────────────
    votes = sum(1 for r in results.values() if r['pred'] == 1)

    binary_pred = 1 if votes >= 1 else 0

    # Confidence — only use models with non-zero confidence (LSTM buffer may be empty)
    active_results = {m: r for m, r in results.items() if r['conf'] > 0.0}
    active_confs   = [r['conf'] for r in active_results.values()]
    active_votes   = sum(1 for r in active_results.values() if r['pred'] == 1)

    if votes == 0:
        final_conf = 0.0
        agreement  = 'none'
    elif len(active_confs) == 0:
        final_conf = 0.0
        agreement  = 'none'
    else:
        # Use only active models for confidence calculation
        if active_votes == len(active_results):
            final_conf = max(active_confs)
            agreement  = 'unanimous'
        elif active_votes >= len(active_results) / 2:
            final_conf = round(sum(active_confs) / len(active_confs), 4)
            agreement  = 'majority'
        else:
            final_conf = max(active_confs)
            agreement  = 'single'

    detecting_models = [m for m, r in results.items() if r['pred'] == 1]
    all_models       = '+'.join(results.keys())
    ml_model_used    = '+'.join(detecting_models) if detecting_models else all_models

    # ── Multiclass identification (only if ATTACK) ────────────
    attack_type = 'BENIGN'

    if binary_pred == 1:
        try:
            if _models['xgb_multi'] is not None:
                # XGBoost n'a pas de feature_names — passe les valeurs numpy directement
                X_multi_vals = X_multi.values
                cls = int(_models['xgb_multi'].predict(X_multi_vals)[0])
            elif _models['rf_multi'] is not None:
                # RF multiclass — passe les valeurs numpy pour éviter le mismatch
                X_multi_vals = X_multi.values
                cls = int(_models['rf_multi'].predict(X_multi)[0])
            else:
                cls = 0

            label = MULTICLASS_LABELS.get(cls, 'Unknown Attack')
            attack_type = 'Unknown Attack' if label == 'BENIGN' else label

        except Exception as e:
            print(f"[ML Engine] Multiclass error: {e}")
            attack_type = 'Unknown Attack'

    return {
        "ml_prediction"  : binary_pred,
        "ml_confidence"  : final_conf,
        "ml_model_used"  : ml_model_used,
        "attack_type"    : attack_type,
        "detection_mode" : "cascade",
        "votes"          : votes,
        "agreement"      : agreement,
        "model_details"  : results,
    }
    # ════════════════════════════════════════════════════════════════
# À AJOUTER dans ml_engine.py (à la fin, avant get_models_status)
# Prédiction vectorisée des 3 modèles (RF + XGBoost + LSTM).
# Le LSTM utilise une fenêtre glissante de SEQUENCE_LEN flux, prédite
# en un seul batch (même technique que agreement_analysis.py).
# ════════════════════════════════════════════════════════════════

def predict_batch(rows: list) -> list:
    """
    Vectorised cascade prediction for a list of flow dicts (RF + XGB + LSTM).

    The LSTM is preserved: flows are scored with a sliding window of
    SEQUENCE_LEN consecutive flows, all windows predicted in one batch.
    The first SEQUENCE_LEN-1 flows can't form a full window -> LSTM vote 0
    for those (RF/XGBoost still cover them).

    Args:
        rows : list of dicts, each mapping feature_name -> value (in order)
    Returns:
        list of dicts (same order) with ml_prediction, ml_confidence,
        ml_model_used, attack_type, detection_mode, votes.
    """
    import numpy as np
    import pandas as pd

    if _models['rf_bin'] is None and _models['xgb_bin'] is None:
        load_models()

    n = len(rows)
    if n == 0:
        return []

    X_bin = pd.DataFrame(
        [{f: r.get(f, 0.0) for f in FEATURE_NAMES} for r in rows]
    )
    X_multi = pd.DataFrame(
        [{f: r.get(f, 0.0) for f in FEATURE_NAMES_MULTI} for r in rows]
    )

    # ── RF + XGBoost binary (whole batch) ─────────────────────
    rf_pred = np.zeros(n, dtype=int); rf_conf = np.zeros(n)
    xgb_pred = np.zeros(n, dtype=int); xgb_conf = np.zeros(n)

    if _models['rf_bin'] is not None:
        rf_pred = _models['rf_bin'].predict(X_bin).astype(int)
        rf_conf = _models['rf_bin'].predict_proba(X_bin)[:, 1]
    if _models['xgb_bin'] is not None:
        xgb_pred = _models['xgb_bin'].predict(X_bin).astype(int)
        xgb_conf = _models['xgb_bin'].predict_proba(X_bin)[:, 1]

    # ── LSTM binary via sliding window (whole batch) ──────────
    lstm_pred = np.zeros(n, dtype=int); lstm_conf = np.zeros(n)
    if _models['lstm_bin'] is not None and _models['scaler_bin'] is not None and n >= SEQUENCE_LEN:
        Xs = _models['scaler_bin'].transform(X_bin.values).astype(np.float32)
        # windows[i] = flows[i .. i+SEQUENCE_LEN-1]; prediction aligns to last flow
        windows = np.stack([Xs[i - SEQUENCE_LEN + 1:i + 1]
                            for i in range(SEQUENCE_LEN - 1, n)])
        probs = _models['lstm_bin'].predict(windows, verbose=0).ravel()
        lstm_conf[SEQUENCE_LEN - 1:] = probs
        lstm_pred[SEQUENCE_LEN - 1:] = (probs >= 0.9).astype(int)

    votes = rf_pred + xgb_pred + lstm_pred          # 0..3
    binary_pred = (votes >= 1).astype(int)

    # ── Multiclass only on detected attacks (whole batch) ─────
    attack_labels = np.array(['BENIGN'] * n, dtype=object)
    atk_idx = np.where(binary_pred == 1)[0]
    if len(atk_idx) and _models['xgb_multi'] is not None:
        cls = _models['xgb_multi'].predict(X_multi.values[atk_idx])
        for j, i in enumerate(atk_idx):
            label = MULTICLASS_LABELS.get(int(cls[j]), 'Unknown Attack')
            attack_labels[i] = 'Unknown Attack' if label == 'BENIGN' else label

    # ── Assemble results ──────────────────────────────────────
    out = []
    for i in range(n):
        detecting = []
        if rf_pred[i] == 1:   detecting.append('RandomForest')
        if xgb_pred[i] == 1:  detecting.append('XGBoost')
        if lstm_pred[i] == 1: detecting.append('LSTM')
        model_used = '+'.join(detecting) if detecting else 'RandomForest+XGBoost+LSTM'

        if binary_pred[i] == 1:
            # confidence = max among models that voted attack (non-zero conf)
            confs = []
            if rf_pred[i] == 1:   confs.append(rf_conf[i])
            if xgb_pred[i] == 1:  confs.append(xgb_conf[i])
            if lstm_pred[i] == 1: confs.append(lstm_conf[i])
            final_conf = round(float(max(confs)), 4) if confs else 0.0
            attack_type = attack_labels[i]
        else:
            final_conf = 0.0
            attack_type = 'BENIGN'

        out.append({
            'ml_prediction' : int(binary_pred[i]),
            'ml_confidence' : final_conf,
            'ml_model_used' : model_used,
            'attack_type'   : attack_type,
            'detection_mode': 'cascade',
            'votes'         : int(votes[i]),
        })
    return out

def get_models_status() -> dict:
    """Return loading status of all models."""
    return {
        "RandomForest_binary"     : _models['rf_bin']    is not None,
        "XGBoost_binary"          : _models['xgb_bin']   is not None,
        "LSTM_binary"             : _models['lstm_bin']  is not None,
        "RandomForest_multiclass" : _models['rf_multi']  is not None,
        "XGBoost_multiclass"      : _models['xgb_multi'] is not None,
        "LSTM_multiclass"         : _models['lstm_multi'] is not None,
    }