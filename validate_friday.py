"""
validate_friday.py — Validate IDS models on real CICIDS2017 Friday PCAP
Project : IDS-KMUTT — Darren Touopi

Friday CICIDS2017 contains: DDoS, PortScan, Botnet + benign traffic.
Reads the CICFlowMeter CSV, runs every flow through the cascade pipeline,
and reports detection counts by attack type and by destination port.
"""

import os
import sys
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ids_kmutt.settings')
django.setup()

import dashboard.ml_engine as engine
engine.load_models()

COLUMN_MAP = {
    'Dst Port':'Destination Port','Flow Duration':'Flow Duration',
    'Total Fwd Packet':'Total Fwd Packets','Total Length of Fwd Packet':'Total Length of Fwd Packets',
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
    'Bwd Packets/s':'Bwd Packets/s','Packet Length Min':'Min Packet Length',
    'Packet Length Max':'Max Packet Length','Packet Length Mean':'Packet Length Mean',
    'Packet Length Variance':'Packet Length Variance','FIN Flag Count':'FIN Flag Count',
    'RST Flag Count':'RST Flag Count','PSH Flag Count':'PSH Flag Count','ACK Flag Count':'ACK Flag Count',
    'URG Flag Count':'URG Flag Count','Down/Up Ratio':'Down/Up Ratio',
    'FWD Init Win Bytes':'Init_Win_bytes_forward','Bwd Init Win Bytes':'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':'act_data_pkt_fwd','Fwd Seg Size Min':'min_seg_size_forward',
    'Active Mean':'Active Mean','Active Std':'Active Std','Active Max':'Active Max',
    'Active Min':'Active Min','Idle Mean':'Idle Mean','Idle Std':'Idle Std',
}


def main(csv_path, limit=None):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)
    df = df.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)

    if limit:
        df = df.head(limit)

    print(f"Total flows: {len(df)}")

    # Build feature matrix once for speed (batch predict)
    X_bin   = pd.DataFrame({f: pd.to_numeric(df[f], errors='coerce').fillna(0.0)
                            for f in engine.FEATURE_NAMES})
    X_multi = X_bin[engine.FEATURE_NAMES_MULTI]

    rf_bin  = engine._models['rf_bin']
    xgb_bin = engine._models['xgb_bin']

    print("Running RF binary...")
    rf_pred = rf_bin.predict(X_bin)
    print("Running XGBoost binary...")
    xgb_pred = xgb_bin.predict(X_bin)

    # Vote: attack if RF or XGB says attack (LSTM needs sequence, skip in batch)
    attack_mask = (rf_pred == 1) | (xgb_pred == 1)
    n_attack = int(attack_mask.sum())
    n_benign = len(df) - n_attack

    print("\n" + "="*55)
    print("BINARY DETECTION RESULTS")
    print("="*55)
    print(f"Total flows : {len(df)}")
    print(f"ATTACK      : {n_attack} ({100*n_attack/len(df):.1f}%)")
    print(f"BENIGN      : {n_benign} ({100*n_benign/len(df):.1f}%)")

    # Multiclass on detected attacks
    if n_attack > 0:
        xgb_multi = engine._models['xgb_multi']
        X_atk = X_multi[attack_mask]
        multi_pred = xgb_multi.predict(X_atk.values)
        labels = pd.Series(multi_pred).map(engine.MULTICLASS_LABELS)
        print(f"\nAttack types identified (multiclass):")
        for label, count in labels.value_counts().items():
            print(f"  {label:<18}: {count}")

    # Detection by destination port (top attacked ports)
    df_atk = df[attack_mask]
    print(f"\nTop attacked destination ports:")
    for port, count in df_atk['Destination Port'].value_counts().head(10).items():
        print(f"  port {int(port):<6}: {count} flows")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/testuser/cicflow_csv/Friday-WorkingHours.pcap_Flow.csv'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(path, limit)
