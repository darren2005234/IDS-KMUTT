"""
agreement_analysis.py — Inter-model agreement on real CICIDS2017 attacks
Project : IDS-KMUTT — Darren Touopi

Measures how often RF, XGBoost and LSTM agree on ATTACK flows, using the
official labelled CICIDS2017 Friday CSVs (ground truth). Produces the
vote distribution (1/3, 2/3, 3/3) and a per-pair agreement matrix.

The LSTM requires a sequence of SEQUENCE_LEN consecutive flows; we feed
flows in order and only score LSTM once its buffer is full.
"""

import os
import sys
import django
import numpy as np
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ids_kmutt.settings')
django.setup()

import dashboard.ml_engine as engine
engine.load_models()

SEQ = engine.SEQUENCE_LEN  # 10

OFFICIAL_MAP = {
    'Destination Port':'Destination Port','Flow Duration':'Flow Duration',
    'Total Fwd Packets':'Total Fwd Packets','Total Length of Fwd Packets':'Total Length of Fwd Packets',
    'Fwd Packet Length Max':'Fwd Packet Length Max','Fwd Packet Length Min':'Fwd Packet Length Min',
    'Fwd Packet Length Mean':'Fwd Packet Length Mean','Fwd Packet Length Std':'Fwd Packet Length Std',
    'Bwd Packet Length Max':'Bwd Packet Length Max','Bwd Packet Length Min':'Bwd Packet Length Min',
    'Bwd Packet Length Mean':'Bwd Packet Length Mean','Flow Bytes/s':'Flow Bytes/s',
    'Flow Packets/s':'Flow Packets/s','Flow IAT Mean':'Flow IAT Mean','Flow IAT Std':'Flow IAT Std',
    'Flow IAT Max':'Flow IAT Max','Flow IAT Min':'Flow IAT Min','Fwd IAT Mean':'Fwd IAT Mean',
    'Fwd IAT Std':'Fwd IAT Std','Fwd IAT Min':'Fwd IAT Min','Bwd IAT Total':'Bwd IAT Total',
    'Bwd IAT Mean':'Bwd IAT Mean','Bwd IAT Std':'Bwd IAT Std','Bwd IAT Max':'Bwd IAT Max',
    'Bwd IAT Min':'Bwd IAT Min','Fwd PSH Flags':'Fwd PSH Flags','Fwd URG Flags':'Fwd URG Flags',
    'Fwd Header Length':'Fwd Header Length','Bwd Header Length':'Bwd Header Length',
    'Bwd Packets/s':'Bwd Packets/s','Min Packet Length':'Min Packet Length',
    'Max Packet Length':'Max Packet Length','Packet Length Mean':'Packet Length Mean',
    'Packet Length Variance':'Packet Length Variance','FIN Flag Count':'FIN Flag Count',
    'RST Flag Count':'RST Flag Count','PSH Flag Count':'PSH Flag Count','ACK Flag Count':'ACK Flag Count',
    'URG Flag Count':'URG Flag Count','Down/Up Ratio':'Down/Up Ratio',
    'Init_Win_bytes_forward':'Init_Win_bytes_forward','Init_Win_bytes_backward':'Init_Win_bytes_backward',
    'act_data_pkt_fwd':'act_data_pkt_fwd','min_seg_size_forward':'min_seg_size_forward',
    'Active Mean':'Active Mean','Active Std':'Active Std','Active Max':'Active Max',
    'Active Min':'Active Min','Idle Mean':'Idle Mean','Idle Std':'Idle Std',
}


def load_X(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    label_col = df.columns[-1]
    y_true = (df[label_col].astype(str).str.strip() != 'BENIGN').astype(int).values
    X = pd.DataFrame({mf: pd.to_numeric(df[oc], errors='coerce').fillna(0.0)
                      for oc, mf in OFFICIAL_MAP.items() if oc in df.columns})
    for f in engine.FEATURE_NAMES:
        if f not in X.columns:
            X[f] = 0.0
    return X[engine.FEATURE_NAMES], y_true


def lstm_predict_sequences(X, scaler, model):
    """Sliding-window LSTM prediction. Flow i scored using flows [i-9 .. i].
    First SEQ-1 flows can't be scored (buffer not full) -> prediction 0."""
    Xs = scaler.transform(X.values).astype(np.float32)
    n = len(Xs)
    preds = np.zeros(n, dtype=int)
    if n < SEQ:
        return preds
    # Build all sliding windows in one batch for speed
    windows = np.stack([Xs[i-SEQ+1:i+1] for i in range(SEQ-1, n)])
    probs = model.predict(windows, verbose=0).ravel()
    preds[SEQ-1:] = (probs >= 0.9).astype(int)
    return preds


def analyse(csv_path, name):
    X, y_true = load_X(csv_path)
    n = len(X)

    rf  = engine._models['rf_bin'].predict(X).astype(int)
    xgb = engine._models['xgb_bin'].predict(X).astype(int)
    lstm = lstm_predict_sequences(X, engine._models['scaler_bin'], engine._models['lstm_bin'])

    # restrict to true-attack flows for agreement on attacks
    atk = (y_true == 1)
    votes = rf[atk] + xgb[atk] + lstm[atk]   # 0..3
    total_atk = int(atk.sum())

    dist = {0:0,1:0,2:0,3:0}
    for v in votes:
        dist[int(v)] += 1

    # pairwise agreement (on attack flows: both say attack)
    rf_a, xgb_a, lstm_a = rf[atk], xgb[atk], lstm[atk]
    def agree(a,b): return int(((a==1)&(b==1)).sum())

    print(f"\n{'='*58}")
    print(f"  {name}  (true attack flows: {total_atk})")
    print(f"{'='*58}")
    print(f"  Detected by 0/3 models : {dist[0]:>8}  ({100*dist[0]/total_atk:5.1f}%)  [missed]")
    print(f"  Detected by 1/3 models : {dist[1]:>8}  ({100*dist[1]/total_atk:5.1f}%)")
    print(f"  Detected by 2/3 models : {dist[2]:>8}  ({100*dist[2]/total_atk:5.1f}%)")
    print(f"  Detected by 3/3 models : {dist[3]:>8}  ({100*dist[3]/total_atk:5.1f}%)  [unanimous]")
    print(f"  ---")
    print(f"  Individual recall on attacks:")
    print(f"    RF      : {100*rf_a.sum()/total_atk:5.1f}%")
    print(f"    XGBoost : {100*xgb_a.sum()/total_atk:5.1f}%")
    print(f"    LSTM    : {100*lstm_a.sum()/total_atk:5.1f}%")
    print(f"  Pairwise agreement (both detect):")
    print(f"    RF & XGB  : {100*agree(rf_a,xgb_a)/total_atk:5.1f}%")
    print(f"    RF & LSTM : {100*agree(rf_a,lstm_a)/total_atk:5.1f}%")
    print(f"    XGB & LSTM: {100*agree(xgb_a,lstm_a)/total_atk:5.1f}%")
    return name, dist, total_atk


if __name__ == '__main__':
    base = '/home/testuser/cicflow_csv'
    files = [
        (f'{base}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv','DDoS'),
        (f'{base}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv','PortScan'),
        (f'{base}/Friday-WorkingHours-Morning.pcap_ISCX.csv','Botnet'),
    ]
    results = []
    for path,name in files:
        if os.path.exists(path):
            results.append(analyse(path,name))

    print(f"\n\n{'='*58}")
    print("SUMMARY — vote distribution on true attacks")
    print(f"{'='*58}")
    print(f"{'Attack':<10}{'1/3':>8}{'2/3':>8}{'3/3':>8}{'missed':>8}")
    for name,dist,tot in results:
        print(f"{name:<10}{100*dist[1]/tot:>7.1f}%{100*dist[2]/tot:>7.1f}%{100*dist[3]/tot:>7.1f}%{100*dist[0]/tot:>7.1f}%")
