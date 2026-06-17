"""
ml_engine.py — ML Model Loading and Inference
Project : IDS-KMUTT v2 — Darren Touopi

v2 changes vs v1:
  - Features : 61 NFStream features (vs 50/47 CICFlowMeter)
  - Scaler   : single scaler_nfstream.joblib (vs separate LSTM scalers)
  - Classes  : 9 classes avec LabelEncoder NFStream
  - Fallback : si modèles v2 absents, charge v1 automatiquement
"""

import os
import joblib
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Model paths ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# v2 paths (NFStream)
RF_BIN_V2       = os.path.join(MODELS_DIR, 'rf_binary_v2.joblib')
XGB_BIN_V2      = os.path.join(MODELS_DIR, 'xgb_binary_v2.joblib')
LSTM_BIN_V2     = os.path.join(MODELS_DIR, 'lstm_binary_v2.keras')
RF_MULTI_V2     = os.path.join(MODELS_DIR, 'rf_multiclass_v2.joblib')
XGB_MULTI_V2    = os.path.join(MODELS_DIR, 'xgb_multiclass_v2.joblib')
LSTM_MULTI_V2   = os.path.join(MODELS_DIR, 'lstm_multiclass_v2.keras')
SCALER_V2       = os.path.join(MODELS_DIR, 'scaler_nfstream.joblib')

# v1 paths (CICFlowMeter — fallback)
RF_BIN_V1       = os.path.join(MODELS_DIR, 'rf_binary_best.joblib')
XGB_BIN_V1      = os.path.join(MODELS_DIR, 'xgb_binary_best.joblib')
LSTM_BIN_V1     = os.path.join(MODELS_DIR, 'lstm_binary_best.keras')
LSTM_BIN_SCALER = os.path.join(MODELS_DIR, 'lstm_scaler.joblib')
RF_MULTI_V1     = os.path.join(MODELS_DIR, 'rf_multiclass_best.joblib')
XGB_MULTI_V1    = os.path.join(MODELS_DIR, 'xgb_multiclass_best.joblib')
LSTM_MULTI_V1   = os.path.join(MODELS_DIR, 'lstm_multiclass_best.keras')
LSTM_MULTI_SCALER = os.path.join(MODELS_DIR, 'lstm_multi_scaler.joblib')

# ── Détection version active ───────────────────────────────────
USE_V2 = os.path.exists(RF_BIN_V2) and os.path.exists(XGB_BIN_V2)

# ── Feature lists ─────────────────────────────────────────────
# v2 : 61 features NFStream — identiques pour binaire ET multiclasse
FEATURE_NAMES_V2 = [
    "src_port", "dst_port", "protocol", "ip_version",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps",
    "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "bidirectional_syn_packets", "bidirectional_cwr_packets",
    "bidirectional_ece_packets", "bidirectional_urg_packets",
    "bidirectional_ack_packets", "bidirectional_psh_packets",
    "bidirectional_rst_packets", "bidirectional_fin_packets",
    "src2dst_syn_packets", "src2dst_cwr_packets", "src2dst_ece_packets",
    "src2dst_urg_packets", "src2dst_ack_packets", "src2dst_psh_packets",
    "src2dst_rst_packets", "src2dst_fin_packets",
    "dst2src_syn_packets", "dst2src_cwr_packets", "dst2src_ece_packets",
    "dst2src_urg_packets", "dst2src_ack_packets", "dst2src_psh_packets",
    "dst2src_rst_packets", "dst2src_fin_packets",
]

# v1 : 50/47 features CICFlowMeter — fallback
FEATURE_NAMES_V1 = [
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
FEATURE_NAMES_MULTI_V1 = [
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

# Aliases actifs selon la version
FEATURE_NAMES       = FEATURE_NAMES_V2 if USE_V2 else FEATURE_NAMES_V1
FEATURE_NAMES_MULTI = FEATURE_NAMES_V2 if USE_V2 else FEATURE_NAMES_MULTI_V1

# ── Class mapping ─────────────────────────────────────────────
# v2 : 9 classes NFStream LabelEncoder (ordre alphabétique sklearn)
MULTICLASS_LABELS_V2 = {
    0: 'BENIGN',
    1: 'Botnet',
    2: 'DDoS',
    3: 'DoS',
    4: 'FTP-Patator',
    5: 'Heartbleed',
    6: 'PortScan',
    7: 'SSH-Patator',
    8: 'Web Attack',
}

# v1 : 8 classes CICFlowMeter
MULTICLASS_LABELS_V1 = {
    0: 'BENIGN',
    1: 'Botnet',
    2: 'Brute Force',
    3: 'DDoS',
    4: 'DoS',
    5: 'PortScan',
    6: 'Rare Attack',
    7: 'Web Attack',
}

MULTICLASS_LABELS = MULTICLASS_LABELS_V2 if USE_V2 else MULTICLASS_LABELS_V1

SEQUENCE_LEN = 10

# ── Model storage ─────────────────────────────────────────────
_models = {
    'rf_bin'      : None,
    'xgb_bin'     : None,
    'lstm_bin'    : None,
    'scaler_bin'  : None,
    'rf_multi'    : None,
    'xgb_multi'   : None,
    'lstm_multi'  : None,
    'scaler_multi': None,
    'version'     : None,
}

_seq_buffer_bin   = []
_seq_buffer_multi = []


def load_models():
    global USE_V2, FEATURE_NAMES, FEATURE_NAMES_MULTI, MULTICLASS_LABELS
    USE_V2 = os.path.exists(RF_BIN_V2) and os.path.exists(XGB_BIN_V2)
    FEATURE_NAMES       = FEATURE_NAMES_V2 if USE_V2 else FEATURE_NAMES_V1
    FEATURE_NAMES_MULTI = FEATURE_NAMES_V2 if USE_V2 else FEATURE_NAMES_MULTI_V1
    MULTICLASS_LABELS   = MULTICLASS_LABELS_V2 if USE_V2 else MULTICLASS_LABELS_V1

    version = "v2 (NFStream)" if USE_V2 else "v1 (CICFlowMeter)"
    print(f"[ML Engine] Loading models — {version}", flush=True)
    _models['version'] = version

    import tensorflow as tf

    # ── RF ────────────────────────────────────────────────────
    rf_bin_path = RF_BIN_V2 if USE_V2 else RF_BIN_V1
    if os.path.exists(rf_bin_path):
        print("[ML Engine] Loading RF binary...")
        _models['rf_bin'] = joblib.load(rf_bin_path)
        print("[ML Engine] RF binary loaded ✅")

    # ── XGBoost ───────────────────────────────────────────────
    xgb_bin_path = XGB_BIN_V2 if USE_V2 else XGB_BIN_V1
    if os.path.exists(xgb_bin_path):
        print("[ML Engine] Loading XGBoost binary...")
        _models['xgb_bin'] = joblib.load(xgb_bin_path)
        print("[ML Engine] XGBoost binary loaded ✅")

    # ── LSTM binaire ──────────────────────────────────────────
    if USE_V2:
        if os.path.exists(LSTM_BIN_V2) and os.path.exists(SCALER_V2):
            print("[ML Engine] Loading LSTM binary (v2)...")
            _models['lstm_bin']   = tf.keras.models.load_model(LSTM_BIN_V2)
            _models['scaler_bin'] = joblib.load(SCALER_V2)
            print("[ML Engine] LSTM binary loaded ✅")
    else:
        if os.path.exists(LSTM_BIN_V1) and os.path.exists(LSTM_BIN_SCALER):
            print("[ML Engine] Loading LSTM binary (v1)...")
            _models['lstm_bin']   = tf.keras.models.load_model(LSTM_BIN_V1)
            _models['scaler_bin'] = joblib.load(LSTM_BIN_SCALER)
            print("[ML Engine] LSTM binary loaded ✅")

    # ── RF multiclasse ────────────────────────────────────────
    rf_multi_path = RF_MULTI_V2 if USE_V2 else RF_MULTI_V1
    if os.path.exists(rf_multi_path):
        print("[ML Engine] Loading RF multiclass...")
        _models['rf_multi'] = joblib.load(rf_multi_path)
        print("[ML Engine] RF multiclass loaded ✅")

    # ── XGBoost multiclasse ───────────────────────────────────
    xgb_multi_path = XGB_MULTI_V2 if USE_V2 else XGB_MULTI_V1
    if os.path.exists(xgb_multi_path):
        print("[ML Engine] Loading XGBoost multiclass...")
        _models['xgb_multi'] = joblib.load(xgb_multi_path)
        print("[ML Engine] XGBoost multiclass loaded ✅")

    # ── LSTM multiclasse ──────────────────────────────────────
    if USE_V2:
        if os.path.exists(LSTM_MULTI_V2) and os.path.exists(SCALER_V2):
            print("[ML Engine] Loading LSTM multiclass (v2)...")
            _models['lstm_multi']   = tf.keras.models.load_model(LSTM_MULTI_V2)
            _models['scaler_multi'] = joblib.load(SCALER_V2)
            print("[ML Engine] LSTM multiclass loaded ✅")
    else:
        if os.path.exists(LSTM_MULTI_V1) and os.path.exists(LSTM_MULTI_SCALER):
            print("[ML Engine] Loading LSTM multiclass (v1)...")
            _models['lstm_multi']   = tf.keras.models.load_model(LSTM_MULTI_V1)
            _models['scaler_multi'] = joblib.load(LSTM_MULTI_SCALER)
            print("[ML Engine] LSTM multiclass loaded ✅")

    print(f"[ML Engine] Ready — {version}", flush=True)


def predict(features: dict, model_name: str = None, mode: str = None) -> dict:
    if _models['rf_bin'] is None and _models['xgb_bin'] is None:
        load_models()

    X_bin   = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURE_NAMES}])
    X_multi = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURE_NAMES_MULTI}])

    results = {}

    # Applique le scaler pour tous les modèles (v2 NFStream)
    if _models['scaler_bin'] is not None:
        X_bin = pd.DataFrame(
            _models['scaler_bin'].transform(X_bin.values),
            columns=X_bin.columns
        )

    # RF
    if _models['rf_bin'] is not None:
        try:
            pred = int(_models['rf_bin'].predict(X_bin)[0])
            conf = float(_models['rf_bin'].predict_proba(X_bin)[0][1])
            results['RandomForest'] = {'pred': pred, 'conf': round(conf, 4)}
        except Exception as e:
            print(f"[ML Engine] RF error: {e}")
            results['RandomForest'] = {'pred': 0, 'conf': 0.0}

    # XGBoost
    if _models['xgb_bin'] is not None:
        try:
            pred = int(_models['xgb_bin'].predict(X_bin)[0])
            conf = float(_models['xgb_bin'].predict_proba(X_bin)[0][1])
            results['XGBoost'] = {'pred': pred, 'conf': round(conf, 4)}
        except Exception as e:
            print(f"[ML Engine] XGB error: {e}")
            results['XGBoost'] = {'pred': 0, 'conf': 0.0}

    # LSTM
    if _models['lstm_bin'] is not None:
        try:
            X_scaled = X_bin.values[0]
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
            print(f"[ML Engine] LSTM error: {e}")
            results['LSTM'] = {'pred': 0, 'conf': 0.0}

    # Voting
    votes = sum(1 for r in results.values() if r['pred'] == 1)
    binary_pred = 1 if votes >= 1 else 0

    active_results = {m: r for m, r in results.items() if r['conf'] > 0.0}
    active_confs   = [r['conf'] for r in active_results.values()]
    active_votes   = sum(1 for r in active_results.values() if r['pred'] == 1)

    if votes == 0 or len(active_confs) == 0:
        final_conf = 0.0
        agreement  = 'none'
    elif active_votes == len(active_results):
        final_conf = max(active_confs)
        agreement  = 'unanimous'
    elif active_votes >= len(active_results) / 2:
        final_conf = round(sum(active_confs) / len(active_confs), 4)
        agreement  = 'majority'
    else:
        final_conf = max(active_confs)
        agreement  = 'single'

    detecting_models = [m for m, r in results.items() if r['pred'] == 1]
    ml_model_used    = '+'.join(detecting_models) if detecting_models else '+'.join(results.keys())

    # Multiclasse
    attack_type = 'BENIGN'
    if binary_pred == 1:
        try:
            if _models['xgb_multi'] is not None:
                X_multi_s = _models['scaler_bin'].transform(X_multi.values) if _models['scaler_bin'] else X_multi.values
                cls = int(_models['xgb_multi'].predict(X_multi_s)[0])
            elif _models['rf_multi'] is not None:
                X_multi_s = _models['scaler_bin'].transform(X_multi.values) if _models['scaler_bin'] else X_multi.values
                cls = int(_models['rf_multi'].predict(X_multi_s)[0])
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


def predict_batch(rows: list) -> list:
    if _models['rf_bin'] is None and _models['xgb_bin'] is None:
        load_models()

    n = len(rows)
    if n == 0:
        return []

    X_bin   = pd.DataFrame([{f: r.get(f, 0.0) for f in FEATURE_NAMES}       for r in rows])
    X_multi = pd.DataFrame([{f: r.get(f, 0.0) for f in FEATURE_NAMES_MULTI} for r in rows])

    # Applique le scaler une seule fois pour tous les modèles
    if _models['scaler_bin'] is not None:
        X_bin_s = pd.DataFrame(
            _models['scaler_bin'].transform(X_bin.values),
            columns=X_bin.columns, index=X_bin.index
        )
    else:
        X_bin_s = X_bin

    rf_pred  = np.zeros(n, dtype=int); rf_conf  = np.zeros(n)
    xgb_pred = np.zeros(n, dtype=int); xgb_conf = np.zeros(n)

    if _models['rf_bin'] is not None:
        rf_pred = _models['rf_bin'].predict(X_bin_s).astype(int)
        rf_conf = _models['rf_bin'].predict_proba(X_bin_s)[:, 1]
    if _models['xgb_bin'] is not None:
        xgb_pred = _models['xgb_bin'].predict(X_bin_s).astype(int)
        xgb_conf = _models['xgb_bin'].predict_proba(X_bin_s)[:, 1]

    lstm_pred = np.zeros(n, dtype=int); lstm_conf = np.zeros(n)
    if _models['lstm_bin'] is not None and n >= SEQUENCE_LEN:
        Xs = X_bin_s.values.astype(np.float32)
        windows = np.stack([Xs[i - SEQUENCE_LEN + 1:i + 1]
                            for i in range(SEQUENCE_LEN - 1, n)])
        probs = _models['lstm_bin'].predict(windows, verbose=0).ravel()
        lstm_conf[SEQUENCE_LEN - 1:] = probs
        lstm_pred[SEQUENCE_LEN - 1:] = (probs >= 0.9).astype(int)

    votes       = rf_pred + xgb_pred + lstm_pred
    binary_pred = (votes >= 1).astype(int)

    # Scale X_multi aussi
    if _models.get('scaler_multi') is not None:
        X_multi_vals = _models['scaler_multi'].transform(X_multi.values)
    elif _models.get('scaler_bin') is not None:
        X_multi_vals = _models['scaler_bin'].transform(X_multi.values)
    else:
        X_multi_vals = X_multi.values

    attack_labels = np.array(['BENIGN'] * n, dtype=object)
    atk_idx = np.where(binary_pred == 1)[0]
    if len(atk_idx) and _models['xgb_multi'] is not None:
        cls = _models['xgb_multi'].predict(X_multi_vals[atk_idx])
        for j, i in enumerate(atk_idx):
            label = MULTICLASS_LABELS.get(int(cls[j]), 'Unknown Attack')
            attack_labels[i] = 'Unknown Attack' if label == 'BENIGN' else label

    out = []
    for i in range(n):
        detecting = []
        if rf_pred[i]   == 1: detecting.append('RandomForest')
        if xgb_pred[i]  == 1: detecting.append('XGBoost')
        if lstm_pred[i] == 1: detecting.append('LSTM')
        model_used = '+'.join(detecting) if detecting else 'RandomForest+XGBoost+LSTM'
        confs = []
        if rf_pred[i]   == 1: confs.append(rf_conf[i])
        if xgb_pred[i]  == 1: confs.append(xgb_conf[i])
        if lstm_pred[i] == 1: confs.append(lstm_conf[i])
        final_conf  = round(float(max(confs)), 4) if confs else 0.0
        attack_type = attack_labels[i] if binary_pred[i] == 1 else 'BENIGN'
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
    return {
        "version"                 : _models.get('version', 'unknown'),
        "RandomForest_binary"     : _models['rf_bin']    is not None,
        "XGBoost_binary"          : _models['xgb_bin']   is not None,
        "LSTM_binary"             : _models['lstm_bin']  is not None,
        "RandomForest_multiclass" : _models['rf_multi']  is not None,
        "XGBoost_multiclass"      : _models['xgb_multi'] is not None,
        "LSTM_multiclass"         : _models['lstm_multi'] is not None,
    }
