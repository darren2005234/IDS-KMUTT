"""
fusion_engine.py — Hybrid Fusion Engine
Project : IDS-KMUTT
Author  : Darren Touopi

4 decision rules combining ML and Snort outputs.
Supports binary and multiclass detection modes.
"""

from .models import Alert


def fuse(src_ip, dst_ip, src_port, dst_port, protocol,
         ml_prediction, ml_confidence, snort_alert, snort_sid,
         ml_model_used, attack_type='BENIGN', detection_mode='binary'):
    """
    Apply fusion rules and persist alert to DB.

    Rules:
        ML=ATTACK + Snort=YES             → THREAT  (BOTH)
        ML=ATTACK + Snort=NO (conf>85%)   → ALERT   (ML_ONLY)
        ML=BENIGN + Snort=YES             → THREAT  (SNORT_ONLY)
        ML=BENIGN + Snort=NO              → BENIGN  (NONE)
    """

    # ── Apply fusion rules ────────────────────────────────────
    if ml_prediction == 1 and snort_alert:
        decision   = "THREAT"
        source_tag = "BOTH"

    elif ml_prediction == 1 and not snort_alert and ml_confidence >= 0.50:
        decision   = "ALERT"
        source_tag = "ML_ONLY"

    elif ml_prediction == 1 and not snort_alert and ml_confidence < 0.50:
        decision   = "BENIGN"
        source_tag = "NONE"

    elif ml_prediction == 0 and snort_alert:
        decision   = "THREAT"
        source_tag = "SNORT_ONLY"

    else:
        decision   = "BENIGN"
        source_tag = "NONE"

    # ── Persist to database ───────────────────────────────────
    alert = Alert.objects.create(
        src_ip         = src_ip,
        dst_ip         = dst_ip,
        src_port       = src_port,
        dst_port       = dst_port,
        protocol       = protocol,
        ml_prediction  = ml_prediction,
        ml_confidence  = ml_confidence,
        ml_model_used  = ml_model_used,
        attack_type    = attack_type,
        detection_mode = detection_mode,
        snort_alert    = snort_alert,
        snort_sid      = snort_sid,
        decision       = decision,
        source_tag     = source_tag,
    )

    return {
        "alert_id"      : alert.id,
        "decision"      : decision,
        "source_tag"    : source_tag,
        "ml_prediction" : ml_prediction,
        "ml_confidence" : ml_confidence,
        "ml_model_used" : ml_model_used,
        "attack_type"   : attack_type,
        "detection_mode": detection_mode,
        "snort_alert"   : snort_alert,
        "snort_sid"     : snort_sid,
    }