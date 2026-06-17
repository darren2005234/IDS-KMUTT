#!/usr/bin/env python3
"""
evaluate_v2.py — Evaluation des modèles v2 (NFStream) sur cicids2017_nfstream_labeled.csv
Projet IDS-KMUTT — Darren Touopi

MODES :
  --day tuesday|wednesday|thursday|friday|all
  --compare-modes : compare cascade vs multiclasse seul

Exemples :
  python evaluate_v2.py --day all --compare-modes
  python evaluate_v2.py --day friday --compare-modes
  python evaluate_v2.py --day tuesday
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_DIR = Path.home() / "ids_kmutt"

# ── Default CSV path ──────────────────────────────────────────────────────
DEFAULT_CSV = Path("~/cicflow_csv/cicids2017_nfstream_labeled.csv").expanduser()

# ── Bornes timestamp par jour (ms Unix) ──────────────────────────────────
DAY_BOUNDS = {
    "monday":    (1499040000000, 1499126399000),
    "tuesday":   (1499126400000, 1499212799000),
    "wednesday": (1499212800000, 1499299199000),
    "thursday":  (1499299200000, 1499385599000),
    "friday":    (1499385600000, 1499471999000),
}

# ── Mapping label → mc_id (ordre alphabétique sklearn LabelEncoder) ───────
LABEL_TO_MCID = {
    "BENIGN":      0,
    "Botnet":      1,
    "DDoS":        2,
    "DoS":         3,
    "FTP-Patator": 4,
    "Heartbleed":  5,
    "PortScan":    6,
    "SSH-Patator": 7,
    "Web Attack":  8,
}

# Classes d'attaques (sans BENIGN et UNLABELED)
ATTACK_CLASSES = [k for k in LABEL_TO_MCID if k != "BENIGN"]

# ── 61 features NFStream ──────────────────────────────────────────────────
SCALER_PATH = Path("~/ids_kmutt/models/scaler_nfstream.joblib").expanduser()

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


# ── Chargement des modèles ────────────────────────────────────────────────
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
    if m.get('lstm_bin') and n >= SEQ:
        # Features déjà scalées dans build_matrix — pas de double scaling
        Xs = X_bin.values.astype(np.float32)
        windows = np.stack([Xs[i-SEQ+1:i+1] for i in range(SEQ-1, n)])
        probs = m['lstm_bin'].predict(windows, verbose=0).ravel()
        lstm[SEQ-1:] = (probs >= 0.9).astype(int)
    return {"rf": rf, "xgb": xgb, "lstm": lstm}


def predict_multiclass(engine, X_mc: pd.DataFrame) -> np.ndarray:
    m = engine._models
    if m.get('xgb_multi'):
        return m['xgb_multi'].predict(X_mc.values).astype(int)
    if m.get('rf_multi'):
        return m['rf_multi'].predict(X_mc.values).astype(int)
    return np.zeros(len(X_mc), dtype=int)


# ── CSV ───────────────────────────────────────────────────────────────────
def load_and_filter(csv_path: str, day: str) -> pd.DataFrame:
    print(f"Chargement {csv_path}...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Filtre par jour via timestamp
    if day != "all" and "bidirectional_first_seen_ms" in df.columns:
        lo, hi = DAY_BOUNDS[day]
        df = df[(df["bidirectional_first_seen_ms"] >= lo) &
                (df["bidirectional_first_seen_ms"] <= hi)]
        print(f"  → {len(df)} flux pour {day}", flush=True)

    # Exclut UNLABELED
    if "Label" in df.columns:
        df = df[df["Label"] != "UNLABELED"].copy()

    return df


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        sys.exit(f"[STOP] {len(missing)} feature(s) absente(s) : {missing[:4]}")
    X = df[FEATURE_NAMES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    # Applique le StandardScaler (modèles entraînés sur données normalisées)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_NAMES, index=X.index)
    else:
        print("[WARN] Scaler non trouvé — features non normalisées", flush=True)
    return X


# ── Métriques ─────────────────────────────────────────────────────────────
def _cfm(y_true, y_pred):
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    recall = tp/(tp+fn) if (tp+fn) else float("nan")
    prec   = tp/(tp+fp) if (tp+fp) else float("nan")
    fpr    = fp/(fp+tn) if (fp+tn) else float("nan")
    f1     = (2*prec*recall/(prec+recall)
              if not (np.isnan(prec) or np.isnan(recall) or prec+recall==0)
              else float("nan"))
    return recall, prec, f1, fpr


# ── Rapport ───────────────────────────────────────────────────────────────
def report(day, attack, y_true, cascade_pred, votes,
           mc_pred, mc_target, compare_modes=False) -> dict:
    tp, fp, fn, tn = _cfm(y_true, cascade_pred)
    has_benign = (tn+fp) > 0
    recall, prec, f1, fpr = _metrics(tp, fp, fn, tn)

    W = 70
    print(f"\n{'═'*W}")
    print(f"  {attack}   [{day} / source=nfstream_v2]")
    print(f"{'─'*W}")
    print(f"  flux : {tp+fn:>7} attaque(s)"
          + (f"   {tn+fp:>7} benin(s)" if has_benign else ""))

    print(f"\n  ┌─ CASCADE  (binaire ≥1 vote → ATTACK)")
    print(f"  │  Recall    : {recall:.4f}   ({tp} / {tp+fn})")
    if has_benign:
        print(f"  │  Precision : {prec:.4f}   F1 : {f1:.4f}   FPR : {fpr:.6f}")

    att_votes = votes[y_true==1]
    if len(att_votes):
        v3 = float(np.mean(att_votes==3)*100)
        v2 = float(np.mean(att_votes==2)*100)
        v1 = float(np.mean(att_votes==1)*100)
        v0 = float(np.mean(att_votes==0)*100)
        print(f"  │  Distribution votes :")
        for k, pct in ((3,v3),(2,v2),(1,v1)):
            bar = '█'*max(0,int(pct/5))
            print(f"  │    {k}/3 : {pct:5.1f}%  {bar}")
        print(f"  │    0/3 (raté) : {v0:.1f}%")
    else:
        v3=v2=v1=v0=float("nan")

    detected = (y_true==1)&(cascade_pred==1)
    typed_ok_pct = unknown_pct = float("nan")
    if mc_pred is not None and detected.sum():
        typed_ok_pct = float(np.mean(mc_pred[detected]==mc_target)*100)
        unknown_pct  = float(np.mean(mc_pred[detected]==0)*100)
        wrong_pct    = 100.0-typed_ok_pct-unknown_pct
        print(f"  │  Typage ({detected.sum()} détectés) :")
        print(f"  │    Correct  ({attack:>15}) : {typed_ok_pct:5.1f}%")
        print(f"  │    Unknown Attack (mc=BENIGN): {unknown_pct:5.1f}%")
        if wrong_pct > 0.5:
            print(f"  │    Mauvaise classe            : {wrong_pct:5.1f}%")
    print(f"  └{'─'*(W-3)}")

    recall_mc = prec_mc = f1_mc = fpr_mc = float("nan")
    rescued = rescued_v3 = rescued_v2 = rescued_v1 = 0

    if compare_modes and mc_pred is not None:
        mc_only = (mc_pred!=0).astype(int)
        tp2,fp2,fn2,tn2 = _cfm(y_true, mc_only)
        recall_mc,prec_mc,f1_mc,fpr_mc = _metrics(tp2,fp2,fn2,tn2)

        rescued_mask = (y_true==1)&(cascade_pred==1)&(mc_only==0)
        rescued = int(rescued_mask.sum())
        delta   = recall - recall_mc

        print(f"\n  ┌─ MULTICLASSE SEUL  (sans couche binaire)")
        print(f"  │  Recall    : {recall_mc:.4f}   ({tp2} / {tp2+fn2})")
        if has_benign:
            print(f"  │  Precision : {prec_mc:.4f}   F1 : {f1_mc:.4f}   FPR : {fpr_mc:.6f}")
        print(f"  └{'─'*(W-3)}")

        print(f"\n  ┌─ VALEUR DU BINAIRE")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "  nan"
        print(f"  │  ΔRecall (cascade − mc_seul) : {delta_str}")
        if tp:
            print(f"  │  Flows rescues : {rescued} ({rescued/max(tp,1)*100:.1f}% des TP cascade)")
        if rescued:
            rv = votes[rescued_mask]
            rescued_v3=int((rv==3).sum())
            rescued_v2=int((rv==2).sum())
            rescued_v1=int((rv==1).sum())
            for k,n in ((3,rescued_v3),(2,rescued_v2),(1,rescued_v1)):
                print(f"  │    {k}/3 : {n:>5} ({n/rescued*100:.0f}%)")
        else:
            print(f"  │  → le multiclasse seul aurait détecté autant que la cascade")
        print(f"  └{'─'*(W-3)}")

    return {
        "day": day, "attack": attack,
        "n_attack": tp+fn, "n_benign": tn+fp,
        "recall": recall, "prec": prec, "f1": f1, "fpr": fpr,
        "v3_pct": v3, "v2_pct": v2, "v1_pct": v1, "v0_pct": v0,
        "typed_ok_pct": typed_ok_pct, "unknown_pct": unknown_pct,
        "recall_mc": recall_mc, "f1_mc": f1_mc,
        "rescued": rescued,
    }


# ── Table résumé ─────────────────────────────────────────────────────────
def _f(v, dec=4):
    return f"{v:.{dec}f}" if not np.isnan(v) else "  nan"
def _p(v):
    return f"{v:5.1f}%" if not np.isnan(v) else "  nan"

def print_summary_table(results, compare_modes=False):
    results = [r for r in results if r["n_attack"]>0]
    if not results:
        print("\n[aucune attaque trouvée]"); return
    W = 112 if compare_modes else 88
    print(f"\n\n{'═'*W}")
    print(f"  RÉSUMÉ — IDS-KMUTT v2  [source: nfstream]")
    print(f"{'─'*W}")
    if compare_modes:
        print(f"  {'Attaque':<18} │{'N_att':>7}│{'Rcl_CAS':>8}│{'F1_CAS':>8}│"
              f"{'FPR_CAS':>9}│{'3/3%':>6}│{'TypedOK':>8}│{'UnknAtt':>8}│"
              f"{'Rcl_MC':>8}│{'F1_MC':>7}│{'ΔRcl':>7}│{'Rescued':>8}")
    else:
        print(f"  {'Attaque':<18} │{'N_att':>7}│{'Recall':>8}│{'Prec.':>8}│"
              f"{'F1':>8}│{'FPR':>9}│{'3/3%':>6}│{'TypedOK':>8}│{'UnknAtt':>8}")
    print(f"{'─'*W}")
    for r in results:
        atk = r["attack"][:18]
        if compare_modes:
            delta = r["recall"]-r["recall_mc"]
            ds = f"{delta:+.4f}" if not np.isnan(delta) else "   nan"
            print(f"  {atk:<18} │{r['n_attack']:>7}│{_f(r['recall']):>8}│"
                  f"{_f(r['f1']):>8}│{_f(r['fpr'],6):>9}│{_p(r['v3_pct']):>6}│"
                  f"{_p(r['typed_ok_pct']):>8}│{_p(r['unknown_pct']):>8}│"
                  f"{_f(r['recall_mc']):>8}│{_f(r['f1_mc']):>7}│{ds:>7}│{r['rescued']:>8}")
        else:
            print(f"  {atk:<18} │{r['n_attack']:>7}│{_f(r['recall']):>8}│"
                  f"{_f(r['prec']):>8}│{_f(r['f1']):>8}│{_f(r['fpr'],6):>9}│"
                  f"{_p(r['v3_pct']):>6}│{_p(r['typed_ok_pct']):>8}│{_p(r['unknown_pct']):>8}")
    print(f"{'═'*W}")
    if compare_modes:
        total = sum(r["rescued"] for r in results)
        valid = [r for r in results if not np.isnan(r["recall"]) and not np.isnan(r["recall_mc"])]
        if valid:
            avg = np.mean([r["recall"]-r["recall_mc"] for r in valid])
            print(f"  Total flows rescues : {total}")
            print(f"  ΔRecall moyen       : {avg:+.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluation IDS-KMUTT v2 (NFStream).")
    ap.add_argument("--day", required=True,
                    choices=list(DAY_BOUNDS.keys())+["all"])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--compare-modes", action="store_true")
    args = ap.parse_args()

    engine = load_engine()
    df = load_and_filter(args.csv, args.day)

    if len(df) == 0:
        sys.exit("[STOP] Aucun flux après filtrage.")

    print(f"Total flux : {len(df)} | Labels : {df['Label'].value_counts().to_dict()}", flush=True)

    X = build_matrix(df)
    y_label = df["Label"].values

    per_model = predict_binary_per_model(engine, X)
    votes     = per_model["rf"] + per_model["xgb"] + per_model["lstm"]
    cascade   = (votes >= 1).astype(int)
    mc_pred   = predict_multiclass(engine, X)

    # Évalue chaque classe d'attaque
    results = []
    for attack in ATTACK_CLASSES:
        mc_target = LABEL_TO_MCID[attack]
        is_attack = (y_label == attack)
        is_benign = (y_label == "BENIGN")
        keep      = is_attack | is_benign
        if not is_attack.sum():
            continue
        y_true = is_attack[keep].astype(int)
        r = report(
            args.day, attack,
            y_true, cascade[keep], votes[keep],
            mc_pred[keep], mc_target,
            args.compare_modes
        )
        results.append(r)

    print_summary_table(results, compare_modes=args.compare_modes)


if __name__ == "__main__":
    main()
