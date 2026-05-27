"""
fusion_engine.py — Hybrid IDS Fusion Engine
Project : IDS-KMUTT
Author  : Darren Touopi

Combines ML model output and Snort alert to produce
a unified decision with a source tag.
"""

from .models import Alert


# ML confidence threshold for ML-only alerts
ML_CONFIDENCE_THRESHOLD = 0.85


def fuse(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: str,
    ml_prediction: int,
    ml_confidence: float,
    snort_alert: bool,
    snort_sid: str = None,
    ml_model_used: str = "RandomForest"
) -> dict:
    """
    Apply the 4 fusion decision rules and persist the alert to DB.

    Rules:
        ML=ATTACK + Snort=YES  → THREAT  (source: BOTH)
        ML=ATTACK + Snort=NO   → ALERT if confidence > 0.85 (source: ML_ONLY)
        ML=BENIGN + Snort=YES  → THREAT  (source: SNORT_ONLY)
        ML=BENIGN + Snort=NO   → BENIGN  (source: NONE)
    """

    # ── Apply fusion rules ──────────────────────────────────
    if ml_prediction == 1 and snort_alert:
        decision   = "THREAT"
        source_tag = "BOTH"

    elif ml_prediction == 1 and not snort_alert:
        if ml_confidence >= ML_CONFIDENCE_THRESHOLD:
            decision   = "ALERT"
            source_tag = "ML_ONLY"
        else:
            decision   = "BENIGN"
            source_tag = "NONE"

    elif ml_prediction == 0 and snort_alert:
        decision   = "THREAT"
        source_tag = "SNORT_ONLY"

    else:
        decision   = "BENIGN"
        source_tag = "NONE"

    # ── Persist to database ─────────────────────────────────
    alert = Alert.objects.create(
        src_ip        = src_ip,
        dst_ip        = dst_ip,
        src_port      = src_port,
        dst_port      = dst_port,
        protocol      = protocol,
        ml_prediction = ml_prediction,
        ml_confidence = round(ml_confidence, 4),
        ml_model_used = ml_model_used,
        snort_alert   = snort_alert,
        snort_sid     = snort_sid,
        decision      = decision,
        source_tag    = source_tag,
    )

    return {
        "alert_id"      : alert.id,
        "decision"      : decision,
        "source_tag"    : source_tag,
        "ml_prediction" : ml_prediction,
        "ml_confidence" : ml_confidence,
        "snort_alert"   : snort_alert,
        "snort_sid"     : snort_sid,
    }