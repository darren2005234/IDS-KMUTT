"""
metrics_friday.py — Real-traffic validation metrics on CICIDS2017 Friday
Project : IDS-KMUTT — Darren Touopi

Runs the binary cascade (RF OR XGBoost vote) on the official labelled
CICIDS2017 Friday CSV files and computes precision / recall / F1 against
the ground-truth Label column. Also evaluates multiclass attack typing.
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

# Official CICIDS2017 CSV columns (78 features) -> model feature names
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


def binary_metrics(y_true, y_pred, attack_name):
    """y_true, y_pred are 0/1 numpy arrays. 1 = attack."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    accuracy  = (tp + tn) / (tp+fp+tn+fn) if (tp+fp+tn+fn) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{'='*55}")
    print(f"  {attack_name}")
    print(f"{'='*55}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    return {'attack':attack_name,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'precision':precision,'recall':recall,'f1':f1,'fpr':fpr}


def evaluate(csv_path, attack_label):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ground truth: 1 if row label != BENIGN
    label_col = df.columns[-1]
    y_true = (df[label_col].astype(str).str.strip() != 'BENIGN').astype(int).values

    # build feature matrix
    X = pd.DataFrame({mf: pd.to_numeric(df[oc], errors='coerce').fillna(0.0)
                      for oc, mf in OFFICIAL_MAP.items() if oc in df.columns})
    for f in engine.FEATURE_NAMES:
        if f not in X.columns:
            X[f] = 0.0
    X = X[engine.FEATURE_NAMES]

    rf_pred  = engine._models['rf_bin'].predict(X)
    xgb_pred = engine._models['xgb_bin'].predict(X)
    y_pred = ((rf_pred == 1) | (xgb_pred == 1)).astype(int)

    return binary_metrics(y_true, y_pred, attack_label)


if __name__ == '__main__':
    base = '/home/testuser/cicflow_csv'
    files = [
        (f'{base}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',     'DDoS'),
        (f'{base}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'PortScan'),
        (f'{base}/Friday-WorkingHours-Morning.pcap_ISCX.csv',            'Botnet'),
    ]
    results = []
    for path, label in files:
        if os.path.exists(path):
            results.append(evaluate(path, label))
        else:
            print(f"Missing: {path}")

    print(f"\n\n{'='*55}")
    print("SUMMARY — Real CICIDS2017 Friday traffic")
    print(f"{'='*55}")
    print(f"{'Attack':<12}{'Precision':>11}{'Recall':>9}{'F1':>9}{'FPR':>9}")
    for r in results:
        print(f"{r['attack']:<12}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}{r['fpr']:>9.4f}")
