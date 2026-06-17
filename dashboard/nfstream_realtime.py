"""
nfstream_realtime.py — Real-time hybrid IDS pipeline (NFStream streaming + Snort)
Project : IDS-KMUTT v2 — Darren Touopi
NFStream stream directement sur eth0. Latence : idle_timeout (5s) au lieu de 30s.
"""

import os
import re
import time
import signal
import logging
import threading
from datetime import datetime
import requests
from nfstream import NFStreamer

API_BATCH_URL  = 'http://127.0.0.1:8000/api/classify_batch/'
BATCH_SIZE     = 500
INTERFACE      = 'eth0'
IDLE_TIMEOUT   = 2
ACTIVE_TIMEOUT = 300
SNORT_ALERT    = '/var/log/snort/snort.alert.fast'
SNORT_REFRESH  = 10

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [nfstream] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

FEATURE_NAMES = [
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

SNORT_RE = re.compile(
    r'^(\d+/\d+-\d+:\d+:\d+)\.\d+.*?\[1:(\d+):\d+\]'
    r'.*?(\d+\.\d+\.\d+\.\d+)(?::\d+)?\s+->\s+(\d+\.\d+\.\d+\.\d+)'
)

_stop = threading.Event()


def clean(v):
    try:
        f = float(v)
        return f if f == f and f not in (float('inf'), float('-inf')) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_snort_alerts(max_lines=20000, max_age_s=90):
    pairs = {}
    now = datetime.now()
    try:
        with open(SNORT_ALERT, 'r', errors='replace') as fh:
            lines = fh.readlines()[-max_lines:]
        for line in lines:
            m = SNORT_RE.match(line)
            if not m:
                continue
            ts_str, sid, src, dst = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                ts = datetime.strptime(f"{now.year}/{ts_str}", "%Y/%m/%d-%H:%M:%S")
                if (now - ts).total_seconds() > max_age_s:
                    continue
            except Exception:
                continue
            pairs[(src, dst)] = sid
    except Exception as e:
        log.error("snort read error: %s", e)
    return pairs


def flush_batch(batch):
    if not batch:
        return 0, 0
    try:
        r = requests.post(API_BATCH_URL, json={'flows': batch}, timeout=60)
        j = r.json()
        return j.get('created', 0), j.get('alerts', 0)
    except Exception as e:
        log.error("batch API error: %s", e)
        return 0, 0


def streaming_thread():
    log.info("NFStream streaming on %s (idle_timeout=%ss)", INTERFACE, IDLE_TIMEOUT)
    snort_pairs     = {}
    last_snort_read = 0
    matched_pairs   = set()
    batch           = []
    total_sent      = 0
    total_alerts    = 0
    window_start    = time.time()
    last_flush      = time.time()

    try:
        streamer = NFStreamer(
            source=INTERFACE,
            statistical_analysis=True,
            splt_analysis=False,
            n_dissections=20,
            idle_timeout=IDLE_TIMEOUT,
            active_timeout=ACTIVE_TIMEOUT,
        )

        for flow in streamer:
            if _stop.is_set():
                break

            now = time.time()
            if now - last_snort_read > SNORT_REFRESH:
                snort_pairs     = read_snort_alerts()
                last_snort_read = now
                orphans = []
                for (s, d), sid in snort_pairs.items():
                    if (s, d) not in matched_pairs:
                        f = {fn: 0.0 for fn in FEATURE_NAMES}
                        f.update({
                            'src_ip': s, 'dst_ip': d,
                            'src_port': 0, 'dst_port': 0,
                            'protocol': 6,
                            'snort_alert': True,
                            'snort_sid': sid,
                        })
                        orphans.append(f)
                if orphans:
                    flush_batch(orphans)
                    log.info("Orphan Snort alerts posted: %d", len(orphans))
                matched_pairs = set()

            feats = {fn: clean(getattr(flow, fn, 0.0)) for fn in FEATURE_NAMES}
            src = str(flow.src_ip)
            dst = str(flow.dst_ip)
            sid         = snort_pairs.get((src, dst))
            snort_alert = sid is not None
            if snort_alert:
                matched_pairs.add((src, dst))

            feats.update({
                'src_ip':      src,
                'dst_ip':      dst,
                'src_port':    int(flow.src_port),
                'dst_port':    int(flow.dst_port),
                'protocol':    int(flow.protocol),
                'snort_alert': snort_alert,
                'snort_sid':   sid,
            })
            batch.append(feats)

            if len(batch) >= BATCH_SIZE or (batch and now - last_flush >= 5):
                sent, alerts  = flush_batch(batch)
                total_sent   += sent
                total_alerts += alerts
                batch         = []
                last_flush    = now

            if now - window_start >= 30:
                log.info("Last 30s → %d flows sent, %d alerts", total_sent, total_alerts)
                total_sent   = 0
                total_alerts = 0
                window_start = now

        if batch:
            flush_batch(batch)

    except Exception as e:
        log.error("NFStream streaming error: %s", e)


def main():
    signal.signal(signal.SIGINT,  lambda *a: _stop.set())
    signal.signal(signal.SIGTERM, lambda *a: _stop.set())
    t = threading.Thread(target=streaming_thread, daemon=True)
    t.start()
    log.info("IDS-KMUTT v2 pipeline running (NFStream streaming + Snort). Ctrl+C to stop.")
    _stop.wait()
    log.info("stopping...")
    time.sleep(3)


if __name__ == '__main__':
    main()
