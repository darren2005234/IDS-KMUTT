"""
test_cicflow_csv.py — Test IDS models on real CICFlowMeter output
Project : IDS-KMUTT
Author  : Darren Touopi

Reads a CICFlowMeter CSV (same format as CICIDS2017 training data),
maps columns to the 50 model features, and runs each flow through the
cascade detection pipeline.
"""

import os
import sys
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ids_kmutt.settings')
django.setup()

import dashboard.ml_engine as engine
engine.load_models()

# ── Map CICFlowMeter V3 column names → model FEATURE_NAMES ────
# CICFlowMeter V3 uses slightly different names than CICIDS2017 CSV
COLUMN_MAP = {
    'Dst Port'                    : 'Destination Port',
    'Flow Duration'               : 'Flow Duration',
    'Total Fwd Packet'            : 'Total Fwd Packets',
    'Total Length of Fwd Packet'  : 'Total Length of Fwd Packets',
    'Fwd Packet Length Max'       : 'Fwd Packet Length Max',
    'Fwd Packet Length Min'       : 'Fwd Packet Length Min',
    'Fwd Packet Length Mean'      : 'Fwd Packet Length Mean',
    'Fwd Packet Length Std'       : 'Fwd Packet Length Std',
    'Bwd Packet Length Max'       : 'Bwd Packet Length Max',
    'Bwd Packet Length Min'       : 'Bwd Packet Length Min',
    'Bwd Packet Length Mean'      : 'Bwd Packet Length Mean',
    'Flow Bytes/s'                : 'Flow Bytes/s',
    'Flow Packets/s'              : 'Flow Packets/s',
    'Flow IAT Mean'               : 'Flow IAT Mean',
    'Flow IAT Std'                : 'Flow IAT Std',
    'Flow IAT Max'                : 'Flow IAT Max',
    'Flow IAT Min'                : 'Flow IAT Min',
    'Fwd IAT Mean'                : 'Fwd IAT Mean',
    'Fwd IAT Std'                 : 'Fwd IAT Std',
    'Fwd IAT Min'                 : 'Fwd IAT Min',
    'Bwd IAT Total'               : 'Bwd IAT Total',
    'Bwd IAT Mean'                : 'Bwd IAT Mean',
    'Bwd IAT Std'                 : 'Bwd IAT Std',
    'Bwd IAT Max'                 : 'Bwd IAT Max',
    'Bwd IAT Min'                 : 'Bwd IAT Min',
    'Fwd PSH Flags'               : 'Fwd PSH Flags',
    'Fwd URG Flags'               : 'Fwd URG Flags',
    'Fwd Header Length'           : 'Fwd Header Length',
    'Bwd Header Length'           : 'Bwd Header Length',
    'Bwd Packets/s'               : 'Bwd Packets/s',
    'Packet Length Min'           : 'Min Packet Length',
    'Packet Length Max'           : 'Max Packet Length',
    'Packet Length Mean'          : 'Packet Length Mean',
    'Packet Length Variance'      : 'Packet Length Variance',
    'FIN Flag Count'              : 'FIN Flag Count',
    'RST Flag Count'              : 'RST Flag Count',
    'PSH Flag Count'              : 'PSH Flag Count',
    'ACK Flag Count'              : 'ACK Flag Count',
    'URG Flag Count'              : 'URG Flag Count',
    'Down/Up Ratio'               : 'Down/Up Ratio',
    'FWD Init Win Bytes'          : 'Init_Win_bytes_forward',
    'Bwd Init Win Bytes'          : 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts'           : 'act_data_pkt_fwd',
    'Fwd Seg Size Min'            : 'min_seg_size_forward',
    'Active Mean'                 : 'Active Mean',
    'Active Std'                  : 'Active Std',
    'Active Max'                  : 'Active Max',
    'Active Min'                  : 'Active Min',
    'Idle Mean'                   : 'Idle Mean',
    'Idle Std'                    : 'Idle Std',
}


def main(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # remove whitespace

    print(f"Loaded {len(df)} flows from {csv_path}")
    print(f"Columns: {len(df.columns)}")

    # Rename columns to model feature names
    df = df.rename(columns=COLUMN_MAP)

    # Counters
    results = {'ATTACK': 0, 'BENIGN': 0}
    attack_types = {}
    detected_flows = []

    for idx, row in df.iterrows():
        features = {f: row.get(f, 0.0) for f in engine.FEATURE_NAMES}
        result = engine.predict(features)

        if result['ml_prediction'] == 1:
            results['ATTACK'] += 1
            at = result['attack_type']
            attack_types[at] = attack_types.get(at, 0) + 1
            detected_flows.append({
                'src'  : row.get('Src IP', '?'),
                'dport': row.get('Destination Port', '?'),
                'type' : at,
                'conf' : result['ml_confidence'],
            })
        else:
            results['BENIGN'] += 1

    # Summary
    print("\n" + "="*60)
    print(f"RESULTS — {os.path.basename(csv_path)}")
    print("="*60)
    print(f"Total flows  : {len(df)}")
    print(f"ATTACK       : {results['ATTACK']}")
    print(f"BENIGN       : {results['BENIGN']}")
    print(f"\nAttack types detected:")
    for at, count in sorted(attack_types.items(), key=lambda x: -x[1]):
        print(f"  {at:<20}: {count}")

    if detected_flows:
        print(f"\nFirst 10 detected attacks:")
        for f in detected_flows[:10]:
            print(f"  {f['src']} → port {f['dport']} | {f['type']} | conf {f['conf']:.2f}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/home/testuser/cicflow_csv/scan_final.pcap_Flow.csv'
    main(csv_path)
