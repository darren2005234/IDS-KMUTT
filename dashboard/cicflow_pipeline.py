"""
cicflow_pipeline.py — Real-time IDS pipeline using CICFlowMeter
Project : IDS-KMUTT — Darren Touopi

Replaces the Zeek/nfstream pipeline. Captures traffic in time windows,
runs each PCAP slice through CICFlowMeter (the same tool that generated
CICIDS2017), then POSTs the extracted flows to the Django API.

Pipeline:
  tcpdump (Ns window) -> CICFlowMeter (pcap -> csv) -> POST /api/classify/
"""

import os
import csv
import time
import glob
import shutil
import logging
import subprocess

import requests

# ── Configuration ─────────────────────────────────────────────
API_URL       = 'http://127.0.0.1:8000/api/classify/'
INTERFACE     = 'eth0'
HOME_NET      = '10.35.111.10'
WINDOW_SEC    = 15                         # capture window length
WORK_DIR      = '/home/testuser/cicflow_live'
PCAP_DIR      = os.path.join(WORK_DIR, 'pcap')
CSV_DIR       = os.path.join(WORK_DIR, 'csv')
CICFLOW_DIR   = '/home/testuser/CICFlowMeter'
CICFLOW_JAR   = os.path.join(CICFLOW_DIR, 'target/CICFlowMeterV3-0.0.4-SNAPSHOT.jar')
CICFLOW_LIB   = os.path.join(CICFLOW_DIR, 'jnetpcap/linux/jnetpcap-1.4.r1425/')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [cicflow_pipeline] %(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

# CICFlowMeter V3 CSV column -> model feature name
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

FEATURE_NAMES = list(COLUMN_MAP.values())


def capture_window(pcap_path):
    """Capture WINDOW_SEC seconds of traffic to pcap_path."""
    cmd = ['timeout', str(WINDOW_SEC),
           'tcpdump', '-i', INTERFACE, '-w', pcap_path,
           'ip', 'and', 'not', 'port', '22']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_cicflowmeter(pcap_path):
    """Convert pcap to CSV via CICFlowMeter. Returns CSV path or None."""
    cmd = [
        'java',
        f'-Djava.library.path={CICFLOW_LIB}',
        '-cp', CICFLOW_JAR,
        'cic.cs.unb.ca.ifm.Cmd',
        pcap_path,
        CSV_DIR + '/'
    ]
    subprocess.run(cmd, cwd=CICFLOW_DIR,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    csv_path = os.path.join(CSV_DIR, os.path.basename(pcap_path) + '_Flow.csv')
    return csv_path if os.path.exists(csv_path) else None


def clean(v):
    try:
        f = float(v)
        return f if f == f and f not in (float('inf'), float('-inf')) else 0.0
    except (TypeError, ValueError):
        return 0.0


def classify_csv(csv_path):
    """Read CICFlowMeter CSV and POST each flow to the API."""
    sent = alerts = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        for row in reader:
            feats = {COLUMN_MAP[k]: clean(row[k])
                     for k in COLUMN_MAP if k in row}
            for fn in FEATURE_NAMES:
                feats.setdefault(fn, 0.0)

            payload = {
                'src_ip'     : row.get('Src IP', '0.0.0.0'),
                'dst_ip'     : row.get('Dst IP', HOME_NET),
                'src_port'   : int(clean(row.get('Src Port', 0))),
                'dst_port'   : int(feats['Destination Port']),
                'protocol'   : 'TCP',
                'snort_alert': False,
                **feats
            }
            try:
                r = requests.post(API_URL, json=payload, timeout=5)
                if r.json().get('decision') in ('THREAT', 'ALERT'):
                    alerts += 1
                sent += 1
            except Exception as e:
                log.error(f"API error: {e}")
    return sent, alerts


def main():
    os.makedirs(PCAP_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(CICFLOW_DIR, 'logs'), exist_ok=True)

    log.info("CICFlowMeter pipeline started")
    log.info(f"Interface={INTERFACE} window={WINDOW_SEC}s API={API_URL}")

    window = 0
    while True:
        try:
            window += 1
            pcap_path = os.path.join(PCAP_DIR, f'w{window:06d}.pcap')

            capture_window(pcap_path)

            if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) < 100:
                continue  # nothing captured

            csv_path = run_cicflowmeter(pcap_path)
            if not csv_path:
                os.remove(pcap_path)
                continue

            sent, alerts = classify_csv(csv_path)
            if sent:
                log.info(f"window {window}: {sent} flows, {alerts} alerts")

            # cleanup slice files
            os.remove(pcap_path)
            os.remove(csv_path)

        except KeyboardInterrupt:
            log.info("Pipeline stopped")
            break
        except Exception as e:
            log.error(f"window error: {e}")
            time.sleep(2)


if __name__ == '__main__':
    main()
