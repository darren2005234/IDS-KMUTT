"""
cicflow_realtime.py — Near-real-time hybrid IDS pipeline (CICFlowMeter + Snort)
Project : IDS-KMUTT — Darren Touopi

Continuous gapless capture: a background tcpdump rotates a PCAP every
WINDOW_SEC seconds (-G); a worker processes each completed file through
CICFlowMeter, reads recent Snort alerts, fuses ML + Snort, and POSTs to
the Django API.

Snort integration: parses /var/log/snort/snort.alert.fast and builds a
set of (src_ip, dst_ip) pairs seen as alerts in the recent window. A flow
is tagged snort_alert=True (with its SID) when its src/dst pair matches.
"""

import os
import re
import csv
import time
import glob
import queue
import signal
import logging
import threading
import subprocess

import requests

# ── Configuration ─────────────────────────────────────────────
API_URL     = 'http://127.0.0.1:8000/api/classify/'
API_BATCH_URL = 'http://127.0.0.1:8000/api/classify_batch/'
BATCH_SIZE    = 500

INTERFACE   = 'eth0'
WINDOW_SEC  = 30
WORK_DIR    = '/home/testuser/cicflow_live'
PCAP_DIR    = os.path.join(WORK_DIR, 'pcap')
CSV_DIR     = os.path.join(WORK_DIR, 'csv')
CICFLOW_DIR = '/home/testuser/CICFlowMeter'
CICFLOW_JAR = os.path.join(CICFLOW_DIR, 'target/CICFlowMeterV3-0.0.4-SNAPSHOT.jar')
CICFLOW_LIB = os.path.join(CICFLOW_DIR, 'jnetpcap/linux/jnetpcap-1.4.r1425/')
BPF_FILTER  = 'ip and not port 22'
SNORT_ALERT = '/var/log/snort/snort.alert.fast'

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [realtime] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

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

# Snort fast-alert line:
# 06/02-16:57:36.308438 [**] [1:538:15] MSG [**] [Class...] [Priority: 3] {TCP} 192.168.10.25:49162 -> 192.168.10.50:139
SNORT_RE = re.compile(
    r'\[\d+:(\d+):\d+\].*?\{[A-Z0-9-]+\}\s+([\d.]+)(?::\d+)?\s+->\s+([\d.]+)(?::\d+)?'
)

_stop = threading.Event()
_work_q = queue.Queue()


def clean(v):
    try:
        f = float(v)
        return f if f == f and f not in (float('inf'), float('-inf')) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_snort_alerts(max_lines=20000):
    """Return dict {(src_ip,dst_ip): sid} from the tail of the Snort fast log."""
    pairs = {}
    if not os.path.exists(SNORT_ALERT):
        return pairs
    try:
        with open(SNORT_ALERT, 'r', errors='ignore') as f:
            lines = f.readlines()[-max_lines:]
        for ln in lines:
            m = SNORT_RE.search(ln)
            if m:
                sid, src, dst = m.group(1), m.group(2), m.group(3)
                pairs[(src, dst)] = sid
    except Exception as e:
        log.error("snort read error: %s", e)
    return pairs


def capture_thread():
    pattern = os.path.join(PCAP_DIR, 'cap_%Y%m%d_%H%M%S.pcap')
    cmd = ['tcpdump', '-i', INTERFACE, '-G', str(WINDOW_SEC),
           '-w', pattern, '-Z', 'root', BPF_FILTER]
    log.info("capture thread started (rotating pcap every %ss)", WINDOW_SEC)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _stop.wait()
    proc.terminate()


def watcher_thread():
    seen = set()
    while not _stop.is_set():
        files = sorted(glob.glob(os.path.join(PCAP_DIR, 'cap_*.pcap')))
        for f in files[:-1]:
            if f not in seen:
                seen.add(f)
                _work_q.put(f)
        time.sleep(2)


def run_cicflowmeter(pcap_path):
    cmd = ['java', f'-Djava.library.path={CICFLOW_LIB}',
           '-cp', CICFLOW_JAR, 'cic.cs.unb.ca.ifm.Cmd', pcap_path, CSV_DIR + '/']
    subprocess.run(cmd, cwd=CICFLOW_DIR,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    csv_path = os.path.join(CSV_DIR, os.path.basename(pcap_path) + '_Flow.csv')
    return csv_path if os.path.exists(csv_path) else None


def classify_csv(csv_path, snort_pairs):
    """Read a CICFlowMeter CSV, batch the flows, POST them to the batch API."""
    sent = alerts = snort_hits = 0
    batch = []
 
    def flush(b):
        nonlocal sent, alerts
        if not b:
            return
        try:
            r = requests.post(API_BATCH_URL, json={'flows': b}, timeout=60)
            j = r.json()
            sent   += j.get('created', 0)
            alerts += j.get('alerts', 0)
        except Exception as e:
            log.error("batch API error: %s", e)
 
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        for row in reader:
            feats = {COLUMN_MAP[k]: clean(row[k]) for k in COLUMN_MAP if k in row}
            for fn in FEATURE_NAMES:
                feats.setdefault(fn, 0.0)
 
            src = row.get('Src IP', '0.0.0.0')
            dst = row.get('Dst IP', '10.35.111.10')
 
            sid = snort_pairs.get((src, dst))
            snort_alert = sid is not None
            if snort_alert:
                snort_hits += 1
 
            feats.update({
                'src_ip': src, 'dst_ip': dst,
                'src_port': int(clean(row.get('Src Port', 0))),
                'dst_port': int(feats['Destination Port']),
                'protocol': 'TCP',
                'snort_alert': snort_alert,
                'snort_sid': sid,
            })
            batch.append(feats)
 
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch = []
 
    flush(batch)   # remaining flows
    return sent, alerts, snort_hits



def processor_thread():
    while not _stop.is_set() or not _work_q.empty():
        try:
            pcap = _work_q.get(timeout=2)
        except queue.Empty:
            continue
        if os.path.getsize(pcap) < 100:
            os.remove(pcap); continue
        csv_path = run_cicflowmeter(pcap)
        if csv_path:
            snort_pairs = read_snort_alerts()
            sent, alerts, snort_hits = classify_csv(csv_path, snort_pairs)
            log.info("%s -> %d flows, %d alerts, %d snort-matched",
                     os.path.basename(pcap), sent, alerts, snort_hits)
            os.remove(csv_path)
        os.remove(pcap)


def main():
    for d in (PCAP_DIR, CSV_DIR, os.path.join(CICFLOW_DIR, 'logs')):
        os.makedirs(d, exist_ok=True)

    signal.signal(signal.SIGINT,  lambda *a: _stop.set())
    signal.signal(signal.SIGTERM, lambda *a: _stop.set())

    threads = [
        threading.Thread(target=capture_thread,   daemon=True),
        threading.Thread(target=watcher_thread,   daemon=True),
        threading.Thread(target=processor_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    log.info("Hybrid near-real-time pipeline running (CICFlowMeter + Snort). Ctrl+C to stop.")
    _stop.wait()
    log.info("stopping...")
    time.sleep(3)


if __name__ == '__main__':
    main()
