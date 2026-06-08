#!/usr/bin/env python3
"""
evaluate.py — Evaluation generique multi-jours de l'IDS hybride en cascade.
Projet IDS-KMUTT — Darren Touopi.

MODES :
  --day tuesday|wednesday|thursday|friday  : evaluation sur un jour
  --day all                                : tous les jours d'un coup (official only)
  --source official|ours
  --compare-modes : compare cascade (binaire >=1 vote) vs multiclasse seul,
                    et montre les flows "rescues par le binaire"

Exemples :
  python evaluate.py --day tuesday --source official --csv ~/cicflow_csv/Tuesday.pcap_ISCX.csv
  python evaluate.py --day friday  --source official --csv ~/cicflow_csv/Friday-DDos.pcap_ISCX.csv --compare-modes
  python evaluate.py --day all     --source official --compare-modes
  python evaluate.py --day tuesday --source ours     --attack "Brute Force" --csv ~/cicflow_csv/ftp_patator_pure.pcap_Flow.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path.home() / "ids_kmutt"

# ##########################################################################
# ## 1. CONFIGURATION                                                    ###
# ##########################################################################

# --- 1a. Jours -> attaques ---------------------------------------------------
DAY_CONFIG = {
    "tuesday": {
        "FTP-Patator": {"canonical": "Brute Force", "mc_id": 2, "label_aliases": ["FTP-Patator"],
                        "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "21"},
        "SSH-Patator": {"canonical": "Brute Force", "mc_id": 2, "label_aliases": ["SSH-Patator"],
                        "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "22"},
    },
    "wednesday": {
        "DoS Hulk":         {"canonical": "DoS", "mc_id": 4, "label_aliases": ["DoS Hulk"],
                             "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "DoS GoldenEye":    {"canonical": "DoS", "mc_id": 4, "label_aliases": ["DoS GoldenEye"],
                             "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "DoS slowloris":    {"canonical": "DoS", "mc_id": 4, "label_aliases": ["DoS slowloris"],
                             "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "DoS Slowhttptest": {"canonical": "DoS", "mc_id": 4, "label_aliases": ["DoS Slowhttptest"],
                             "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "Heartbleed":       {"canonical": "Rare Attack", "mc_id": 6, "label_aliases": ["Heartbleed"],
                             "attacker": "172.16.0.1", "victims": "192.168.10.51", "dport": "444"},
    },
    "thursday": {
        "Web Attack Brute Force": {"canonical": "Web Attack", "mc_id": 7,
                                   "label_aliases": ["Web Attack \x96 Brute Force",
                                                     "Web Attack - Brute Force", "Web Attack Brute Force"],
                                   "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "Web Attack XSS":         {"canonical": "Web Attack", "mc_id": 7,
                                   "label_aliases": ["Web Attack \x96 XSS",
                                                     "Web Attack - XSS", "Web Attack XSS"],
                                   "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "Web Attack Sql Injection": {"canonical": "Web Attack", "mc_id": 7,
                                   "label_aliases": ["Web Attack \x96 Sql Injection",
                                                     "Web Attack - Sql Injection", "Web Attack Sql Injection"],
                                   "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "Infiltration":           {"canonical": "Rare Attack", "mc_id": 6,
                                   "label_aliases": ["Infiltration"],
                                   "attacker": "192.168.10.8", "victims": "", "dport": ""},
    },
    "friday": {
        "DDoS":     {"canonical": "DDoS",     "mc_id": 3, "label_aliases": ["DDoS"],
                     "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": "80"},
        "PortScan": {"canonical": "PortScan", "mc_id": 5, "label_aliases": ["PortScan"],
                     "attacker": "172.16.0.1", "victims": "192.168.10.50", "dport": ""},
        "Bot":      {"canonical": "Botnet",   "mc_id": 1, "label_aliases": ["Bot", "Botnet"],
                     "attacker": "205.174.165.73,172.16.0.1",
                     "victims": "192.168.10.15,192.168.10.9,192.168.10.14,192.168.10.5,192.168.10.8",
                     "dport": ""},
    },
}

# --- 1b. CSVs officiels pour --day all --------------------------------------
# Ajoute les chemins au fur et a mesure que tu telecharges les fichiers.
# Plusieurs entrees par jour OK (ex: vendredi = 3 fichiers).
# Mets None si pas encore telecharge -> skipped avec un message.
OFFICIAL_CSVS_ALL = [
    ("friday",    "~/cicflow_csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("friday",    "~/cicflow_csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("friday",    "~/cicflow_csv/Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("tuesday",   "~/cicflow_csv/Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("wednesday", "~/cicflow_csv/Wednesday-workingHours.pcap_ISCX.csv"),
    ("thursday",  "~/cicflow_csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("thursday",  "~/cicflow_csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
]

# --- 1c. Features -----------------------------------------------------------
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
_DROP = ["Fwd Packet Length Std", "Bwd Packet Length Mean", "Idle Mean"]
assert [f for f in FEATURE_NAMES if f not in _DROP] == FEATURE_NAMES_MULTI

OFFICIAL_MAP = {f: f for f in FEATURE_NAMES}
OURS_MAP = {
    'Dst Port':'Destination Port','Flow Duration':'Flow Duration',
    'Total Fwd Packet':'Total Fwd Packets',
    'Total Length of Fwd Packet':'Total Length of Fwd Packets',
    'Fwd Packet Length Max':'Fwd Packet Length Max',
    'Fwd Packet Length Min':'Fwd Packet Length Min',
    'Fwd Packet Length Mean':'Fwd Packet Length Mean',
    'Fwd Packet Length Std':'Fwd Packet Length Std',
    'Bwd Packet Length Max':'Bwd Packet Length Max',
    'Bwd Packet Length Min':'Bwd Packet Length Min',
    'Bwd Packet Length Mean':'Bwd Packet Length Mean',
    'Flow Bytes/s':'Flow Bytes/s','Flow Packets/s':'Flow Packets/s',
    'Flow IAT Mean':'Flow IAT Mean','Flow IAT Std':'Flow IAT Std',
    'Flow IAT Max':'Flow IAT Max','Flow IAT Min':'Flow IAT Min',
    'Fwd IAT Mean':'Fwd IAT Mean','Fwd IAT Std':'Fwd IAT Std',
    'Fwd IAT Min':'Fwd IAT Min','Bwd IAT Total':'Bwd IAT Total',
    'Bwd IAT Mean':'Bwd IAT Mean','Bwd IAT Std':'Bwd IAT Std',
    'Bwd IAT Max':'Bwd IAT Max','Bwd IAT Min':'Bwd IAT Min',
    'Fwd PSH Flags':'Fwd PSH Flags','Fwd URG Flags':'Fwd URG Flags',
    'Fwd Header Length':'Fwd Header Length',
    'Bwd Header Length':'Bwd Header Length','Bwd Packets/s':'Bwd Packets/s',
    'Packet Length Min':'Min Packet Length','Packet Length Max':'Max Packet Length',
    'Packet Length Mean':'Packet Length Mean',
    'Packet Length Variance':'Packet Length Variance',
    'FIN Flag Count':'FIN Flag Count','RST Flag Count':'RST Flag Count',
    'PSH Flag Count':'PSH Flag Count','ACK Flag Count':'ACK Flag Count',
    'URG Flag Count':'URG Flag Count','Down/Up Ratio':'Down/Up Ratio',
    'FWD Init Win Bytes':'Init_Win_bytes_forward',
    'Bwd Init Win Bytes':'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':'act_data_pkt_fwd',
    'Fwd Seg Size Min':'min_seg_size_forward',
    'Active Mean':'Active Mean','Active Std':'Active Std',
    'Active Max':'Active Max','Active Min':'Active Min',
    'Idle Mean':'Idle Mean','Idle Std':'Idle Std',
}
RENAME = {"official": OFFICIAL_MAP, "ours": OURS_MAP}

LABEL_COL_CANDIDATES = ["Label", " Label", "label"]
SRC_IP_CANDIDATES    = ["Src IP", "Source IP", "src_ip"]
DST_IP_CANDIDATES    = ["Dst IP", "Destination IP", "dst_ip"]
DPORT_CANDIDATES     = ["Dst Port", "Destination Port", "dst_port"]


# ##########################################################################
# ## 2. MODELES                                                          ###
# ##########################################################################

def load_engine():
    sys.path.insert(0, str(PROJECT_DIR))
    from dashboard import ml_engine
    ml_engine.load_models()
    return ml_engine


def predict_binary_per_model(engine, X_bin: pd.DataFrame) -> dict:
    m = engine._models
    rf  = m['rf_bin'].predict(X_bin).astype(int)
    xgb = m['xgb_bin'].predict(X_bin).astype(int)
    n   = len(X_bin)
    SEQ = engine.SEQUENCE_LEN
    lstm = np.zeros(n, dtype=int)
    if m.get('lstm_bin') is not None and m.get('scaler_bin') is not None and n >= SEQ:
        Xs = m['scaler_bin'].transform(X_bin.values).astype(np.float32)
        windows = np.stack([Xs[i - SEQ + 1:i + 1] for i in range(SEQ - 1, n)])
        probs = m['lstm_bin'].predict(windows, verbose=0).ravel()
        lstm[SEQ - 1:] = (probs >= 0.9).astype(int)
    return {"rf": rf, "xgb": xgb, "lstm": lstm}


def predict_multiclass(engine, X_mc_df: pd.DataFrame) -> np.ndarray:
    m = engine._models
    if m.get('xgb_multi') is not None:
        return m['xgb_multi'].predict(X_mc_df.values).astype(int)
    if m.get('rf_multi') is not None:
        return m['rf_multi'].predict(X_mc_df).astype(int)
    return np.zeros(len(X_mc_df), dtype=int)


# ##########################################################################
# ## 3. LOGIQUE GENERIQUE                                                ###
# ##########################################################################

def _first_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_matrices(df: pd.DataFrame, source: str):
    ren = {k: v for k, v in RENAME.get(source, {}).items() if k in df.columns}
    d = df.rename(columns=ren)
    missing = [f for f in FEATURE_NAMES if f not in d.columns]
    if missing:
        sys.exit(f"[STOP] {len(missing)} feature(s) absente(s) (source={source}). "
                 f"Ex : {missing[:4]}")
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0)
    return d[FEATURE_NAMES].astype(float), d[FEATURE_NAMES_MULTI].astype(float)


def ground_truth_official(df, day):
    """y_canon par flux a partir de la colonne Label officielle.
    Matching tolerant : on cherche les mots-cles dans le label (gere les
    caracteres bizarres de CICIDS2017 comme 'Web Attack <0x96> XSS')."""
    label_col = _first_col(df, LABEL_COL_CANDIDATES)
    if label_col is None:
        sys.exit(f"[STOP] Colonne Label introuvable ({LABEL_COL_CANDIDATES}).")
    labels = df[label_col].astype(str).str.strip()

    # Construit (mots-cles requis -> canonical) depuis les alias.
    # Ex: "Web Attack \x96 XSS" -> on ne garde que les mots alphanum : ['web','attack','xss']
    import re
    def keywords(s):
        return [w for w in re.split(r'[^A-Za-z0-9]+', s.lower()) if w]

    rules = []  # (set_de_mots_cles, canonical)
    for cfg in DAY_CONFIG[day].values():
        for al in cfg["label_aliases"]:
            kw = set(keywords(al))
            if kw:
                rules.append((kw, cfg["canonical"]))

    def classify(x):
        if x.upper() == "BENIGN":
            return "BENIGN"
        xw = set(keywords(x))
        # le label matche une regle si tous les mots-cles de la regle sont presents
        for kw, canon in rules:
            if kw.issubset(xw):
                return canon
        return "OTHER"

    return labels.map(classify).values


def ground_truth_ours(df, attack, attacker, victims, dport):
    sc = _first_col(df, SRC_IP_CANDIDATES)
    dc = _first_col(df, DST_IP_CANDIDATES)
    if sc is None or dc is None:
        sys.exit("[STOP] Colonnes IP introuvables.")
    src = df[sc].astype(str); dst = df[dc].astype(str)
    att = set(a.strip() for a in attacker.split(",")) if attacker else set()
    vic = set(v.strip() for v in victims.split(","))  if victims  else set()
    if att and vic:
        mask = (src.isin(att) & dst.isin(vic)) | (src.isin(vic) & dst.isin(att))
    elif vic:
        mask = dst.isin(vic) | src.isin(vic)
    elif att:
        mask = src.isin(att) | dst.isin(att)
    else:
        sys.exit("[STOP] --source ours exige au moins --victims ou --attacker.")
    if dport:
        pc = _first_col(df, DPORT_CANDIDATES)
        if pc is not None:
            mask = mask & (pd.to_numeric(df[pc], errors='coerce')
                           .fillna(-1).astype(int) == int(dport))
    return np.where(mask.values, attack, "BENIGN")


def _cfm(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    prec   = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr    = fp / (fp + tn) if (fp + tn) else float("nan")
    f1     = (2 * prec * recall / (prec + recall)
              if not (np.isnan(prec) or np.isnan(recall) or prec + recall == 0)
              else float("nan"))
    return recall, prec, f1, fpr


# ##########################################################################
# ## 4. RAPPORT (un seul appel par classe canonique)                    ###
# ##########################################################################

def report(day, source, attack, y_true, cascade_pred, votes,
           mc_pred, mc_target, compare_modes=False) -> dict:
    """Affiche les métriques et retourne un dict pour la table résumé."""
    tp, fp, fn, tn = _cfm(y_true, cascade_pred)
    has_benign = (tn + fp) > 0
    recall, prec, f1, fpr = _metrics(tp, fp, fn, tn)

    W = 70
    print(f"\n{'═' * W}")
    print(f"  {attack}   [{day} / source={source}]")
    print(f"{'─' * W}")
    print(f"  flux : {tp+fn:>7} attaque(s)"
          + (f"   {tn+fp:>7} benin(s)" if has_benign else ""))

    # ── Cascade ──────────────────────────────────────────────────────────
    print(f"\n  ┌─ CASCADE  (binaire ≥1 vote → ATTACK)")
    print(f"  │  Recall    : {recall:.4f}   ({tp} / {tp+fn})")
    if has_benign:
        print(f"  │  Precision : {prec:.4f}   F1 : {f1:.4f}   FPR : {fpr:.6f}")
    else:
        print(f"  │  (P/F1/FPR non calculables : pas de flux benin dans ce CSV)")

    att_votes = votes[y_true == 1]
    v3 = float(np.mean(att_votes == 3) * 100) if len(att_votes) else float("nan")
    v2 = float(np.mean(att_votes == 2) * 100) if len(att_votes) else float("nan")
    v1 = float(np.mean(att_votes == 1) * 100) if len(att_votes) else float("nan")
    v0 = float(np.mean(att_votes == 0) * 100) if len(att_votes) else float("nan")
    if len(att_votes):
        print(f"  │  Distribution votes :")
        for k, pct in ((3, v3), (2, v2), (1, v1)):
            bar = '█' * max(0, int(pct / 5))
            print(f"  │    {k}/3 : {pct:5.1f}%  {bar}")
        print(f"  │    0/3 (raté) : {v0:.1f}%")

    # Typage multiclasse
    detected = (y_true == 1) & (cascade_pred == 1)
    typed_ok_pct = float("nan")
    unknown_pct  = float("nan")
    if mc_pred is not None and detected.sum():
        typed_ok_pct = float(np.mean(mc_pred[detected] == mc_target) * 100)
        unknown_pct  = float(np.mean(mc_pred[detected] == 0) * 100)
        wrong_pct    = 100.0 - typed_ok_pct - unknown_pct
        print(f"  │  Typage ({detected.sum()} détectés) :")
        print(f"  │    Correct  ({attack:>15}) : {typed_ok_pct:5.1f}%")
        print(f"  │    Unknown Attack (mc=BENIGN): {unknown_pct:5.1f}%  ← rescues binaire en prod")
        if wrong_pct > 0.5:
            print(f"  │    Mauvaise classe            : {wrong_pct:5.1f}%")
    print(f"  └{'─' * (W - 3)}")

    # ── Compare modes ────────────────────────────────────────────────────
    recall_mc = prec_mc = f1_mc = fpr_mc = float("nan")
    rescued = 0
    rescued_v3 = rescued_v2 = rescued_v1 = 0

    if compare_modes and mc_pred is not None:
        mc_only = (mc_pred != 0).astype(int)
        tp2, fp2, fn2, tn2 = _cfm(y_true, mc_only)
        recall_mc, prec_mc, f1_mc, fpr_mc = _metrics(tp2, fp2, fn2, tn2)

        rescued_mask = (y_true == 1) & (cascade_pred == 1) & (mc_only == 0)
        rescued = int(rescued_mask.sum())
        delta   = recall - recall_mc

        print(f"\n  ┌─ MULTICLASSE SEUL  (sans couche binaire)")
        print(f"  │  Recall    : {recall_mc:.4f}   ({tp2} / {tp2+fn2})")
        if has_benign:
            print(f"  │  Precision : {prec_mc:.4f}   F1 : {f1_mc:.4f}   FPR : {fpr_mc:.6f}")
        print(f"  └{'─' * (W - 3)}")

        print(f"\n  ┌─ VALEUR DU BINAIRE")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "  nan"
        print(f"  │  ΔRecall (cascade − mc_seul) : {delta_str}")
        if tp:
            print(f"  │  Flows rescues par le binaire : {rescued} "
                  f"({rescued/max(tp,1)*100:.1f}% des TP cascade)")
        if rescued:
            rv = votes[rescued_mask]
            rescued_v3 = int((rv == 3).sum())
            rescued_v2 = int((rv == 2).sum())
            rescued_v1 = int((rv == 1).sum())
            print(f"  │  Distribution votes des flows rescues :")
            for k, n in ((3, rescued_v3), (2, rescued_v2), (1, rescued_v1)):
                print(f"  │    {k}/3 : {n:>5} ({n/rescued*100:.0f}%)")
            print(f"  │  → sans binaire, ces {rescued} attaques = BENIGN non détecté")
        else:
            print(f"  │  → le multiclasse seul aurait détecté autant que la cascade")
        print(f"  └{'─' * (W - 3)}")

    return {
        "day": day, "source": source, "attack": attack,
        "n_attack": tp + fn, "n_benign": tn + fp,
        "recall": recall, "prec": prec, "f1": f1, "fpr": fpr,
        "v3_pct": v3, "v2_pct": v2, "v1_pct": v1, "v0_pct": v0,
        "typed_ok_pct": typed_ok_pct, "unknown_pct": unknown_pct,
        "recall_mc": recall_mc, "f1_mc": f1_mc, "fpr_mc": fpr_mc,
        "rescued": rescued,
        "rescued_v3": rescued_v3, "rescued_v2": rescued_v2, "rescued_v1": rescued_v1,
    }


# ##########################################################################
# ## 5. TABLE RÉSUMÉ                                                     ###
# ##########################################################################

def _f(v, dec=4):
    return f"{v:.{dec}f}" if not np.isnan(v) else "  nan"

def _p(v):
    return f"{v:5.1f}%" if not np.isnan(v) else "  nan"


def print_summary_table(results: list, compare_modes: bool = False):
    results = [r for r in results if r["n_attack"] > 0]
    if not results:
        print("\n[aucune attaque trouvée — vérifie le filtre IP/port]")
        return
    W = 112 if compare_modes else 88
    print(f"\n\n{'═' * W}")
    print(f"  RÉSUMÉ — IDS-KMUTT  [source: {results[0]['source']}]")
    print(f"{'─' * W}")
    if compare_modes:
        print(f"  {'Attaque':<18} │{'N_att':>7}│"
              f"{'Rcl_CAS':>8}│{'F1_CAS':>8}│{'FPR_CAS':>9}│"
              f"{'3/3%':>6}│{'TypedOK':>8}│{'UnknAtt':>8}│"
              f"{'Rcl_MC':>8}│{'F1_MC':>7}│{'ΔRcl':>7}│{'Rescued':>8}")
    else:
        print(f"  {'Attaque':<18} │{'N_att':>7}│"
              f"{'Recall':>8}│{'Prec.':>8}│{'F1':>8}│{'FPR':>9}│"
              f"{'3/3%':>6}│{'TypedOK':>8}│{'UnknAtt':>8}")
    print(f"{'─' * W}")

    prev_day = None
    for r in results:
        if r["day"] != prev_day:
            if prev_day is not None:
                print(f"{'·' * W}")
            prev_day = r["day"]
        atk = r["attack"][:18]
        if compare_modes:
            delta = r["recall"] - r["recall_mc"]
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "   nan"
            print(f"  {atk:<18} │{r['n_attack']:>7}│"
                  f"{_f(r['recall']):>8}│{_f(r['f1']):>8}│{_f(r['fpr'],6):>9}│"
                  f"{_p(r['v3_pct']):>6}│{_p(r['typed_ok_pct']):>8}│{_p(r['unknown_pct']):>8}│"
                  f"{_f(r['recall_mc']):>8}│{_f(r['f1_mc']):>7}│{delta_s:>7}│{r['rescued']:>8}")
        else:
            print(f"  {atk:<18} │{r['n_attack']:>7}│"
                  f"{_f(r['recall']):>8}│{_f(r['prec']):>8}│{_f(r['f1']):>8}│{_f(r['fpr'],6):>9}│"
                  f"{_p(r['v3_pct']):>6}│{_p(r['typed_ok_pct']):>8}│{_p(r['unknown_pct']):>8}")

    print(f"{'═' * W}")
    if compare_modes:
        total_rescued = sum(r["rescued"] for r in results)
        valid = [r for r in results if not np.isnan(r["recall"]) and not np.isnan(r["recall_mc"])]
        if valid:
            avg_delta = np.mean([r["recall"] - r["recall_mc"] for r in valid])
            print(f"  Total flows rescues par le binaire : {total_rescued}")
            print(f"  ΔRecall moyen (cascade − mc_seul)  : {avg_delta:+.4f}")
    print()


# ##########################################################################
# ## 6. EVALUATION D'UN SEUL FICHIER (helper mutualisé)                 ###
# ##########################################################################

def run_file(day, csv_path, source, engine, compare_modes,
             attack_arg="", attacker_arg="", victims_arg="", dport_arg=""):
    """Charge un CSV, prédit, et génère les rapports. Retourne la liste des dicts."""
    df = load_csv(csv_path)
    X_bin, X_mc = build_matrices(df, source)

    if source == "official":
        y_canon = ground_truth_official(df, day)
    else:
        att, vic, dp = attacker_arg, victims_arg, dport_arg
        if not (att or vic):
            for cfg in DAY_CONFIG[day].values():
                if cfg["canonical"] == attack_arg:
                    att = att or cfg.get("attacker", "")
                    vic = vic or cfg.get("victims",  "")
                    dp  = dp  or cfg.get("dport",    "")
                    break
            if att or vic:
                print(f"[info] IPs auto : attacker={att or '-'} "
                      f"victims={vic or '-'} dport={dp or '-'}")
        y_canon = ground_truth_ours(df, attack_arg, att, vic, dp)

    per_model = predict_binary_per_model(engine, X_bin)
    votes = per_model["rf"] + per_model["xgb"] + per_model["lstm"]
    cascade_pred = (votes >= 1).astype(int)
    mc_pred = predict_multiclass(engine, X_mc)

    canon_classes = (sorted({c["canonical"] for c in DAY_CONFIG[day].values()})
                     if source == "official" else [attack_arg])

    results = []
    for canon in canon_classes:
        is_this  = (y_canon == canon)
        is_benign= (y_canon == "BENIGN")
        keep     = is_this | is_benign
        y_true   = is_this[keep].astype(int)
        y_pred   = cascade_pred[keep]
        v        = votes[keep]
        mc       = mc_pred[keep] if mc_pred is not None else None
        mc_target= next((c["mc_id"] for c in DAY_CONFIG[day].values()
                         if c["canonical"] == canon), -1)
        r = report(day, source, canon, y_true, y_pred, v, mc, mc_target, compare_modes)
        results.append(r)
    return results


# ##########################################################################
# ## 7. MAIN                                                             ###
# ##########################################################################

def main():
    ap = argparse.ArgumentParser(description="Evaluation generique IDS hybride en cascade.")
    ap.add_argument("--day",    required=True,
                    choices=sorted(DAY_CONFIG) + ["all"])
    ap.add_argument("--source", required=True, choices=["official", "ours"])
    ap.add_argument("--csv",    default="", help="Chemin CSV (inutile avec --day all)")
    ap.add_argument("--attack",   default="")
    ap.add_argument("--attacker", default="")
    ap.add_argument("--victims",  default="")
    ap.add_argument("--dport",    default="")
    ap.add_argument("--compare-modes", action="store_true",
                    help="Compare CASCADE vs MULTICLASSE SEUL + flows rescues par le binaire")
    args = ap.parse_args()

    if args.day == "all" and args.source == "ours":
        sys.exit("[STOP] --day all est supporte uniquement avec --source official.\n"
                 "       Pour 'ours', lance une evaluation par jour manuellement.")

    # Charge les modeles UNE SEULE FOIS (important pour --day all)
    engine = load_engine()

    if args.day == "all":
        all_results = []
        skipped = []
        for day, csv_path in OFFICIAL_CSVS_ALL:
            if csv_path is None:
                skipped.append(f"{day} (non configure)")
                continue
            p = Path(csv_path).expanduser()
            if not p.exists():
                skipped.append(f"{day} ({p.name})")
                continue
            print(f"\n{'▶' * 3}  {day.upper()}  —  {p.name}")
            results = run_file(day, str(p), "official", engine, args.compare_modes)
            all_results.extend(results)

        if skipped:
            print(f"\n[skip] {len(skipped)} fichier(s) non disponible(s) : "
                  + ", ".join(skipped))

        print_summary_table(all_results, compare_modes=args.compare_modes)

    else:
        if not args.csv:
            sys.exit("[STOP] --csv est requis pour un jour specifique.")
        if args.source == "ours" and not args.attack:
            sys.exit("[STOP] --source ours exige --attack \"<Nom canonique>\".")

        results = run_file(
            args.day, args.csv, args.source, engine, args.compare_modes,
            attack_arg=args.attack,
            attacker_arg=args.attacker,
            victims_arg=args.victims,
            dport_arg=args.dport,
        )
        print_summary_table(results, compare_modes=args.compare_modes)


if __name__ == "__main__":
    main()