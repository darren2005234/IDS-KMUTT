"""
SHAP Analysis — IDS-KMUTT v2 (NFStream)
Explicabilité des décisions XGBoost multiclasse par classe d'attaque
Darren Touopi — KMUTT Bangkok 2026
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR = Path.home() / 'ids_kmutt'
DATA_DIR    = Path.home() / 'data'
MODEL_DIR   = PROJECT_DIR / 'models'
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── Features ──────────────────────────────────────────────────
FEATURE_NAMES = [
    'src_port', 'dst_port', 'protocol', 'ip_version',
    'bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes',
    'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
    'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes',
    'bidirectional_min_ps', 'bidirectional_mean_ps',
    'bidirectional_stddev_ps', 'bidirectional_max_ps',
    'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps', 'dst2src_max_ps',
    'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
    'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms',
    'src2dst_min_piat_ms', 'src2dst_mean_piat_ms',
    'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    'dst2src_min_piat_ms', 'dst2src_mean_piat_ms',
    'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms',
    'bidirectional_syn_packets', 'bidirectional_cwr_packets',
    'bidirectional_ece_packets', 'bidirectional_urg_packets',
    'bidirectional_ack_packets', 'bidirectional_psh_packets',
    'bidirectional_rst_packets', 'bidirectional_fin_packets',
    'src2dst_syn_packets', 'src2dst_cwr_packets', 'src2dst_ece_packets',
    'src2dst_urg_packets', 'src2dst_ack_packets', 'src2dst_psh_packets',
    'src2dst_rst_packets', 'src2dst_fin_packets',
    'dst2src_syn_packets', 'dst2src_cwr_packets', 'dst2src_ece_packets',
    'dst2src_urg_packets', 'dst2src_ack_packets', 'dst2src_psh_packets',
    'dst2src_rst_packets', 'dst2src_fin_packets',
]

LABEL_MAP = {
    0: 'BENIGN', 1: 'Botnet', 2: 'DDoS', 3: 'DoS',
    4: 'FTP-Patator', 5: 'Heartbleed', 6: 'PortScan',
    7: 'SSH-Patator', 8: 'Web Attack'
}
ATTACK_CLASSES = [v for v in LABEL_MAP.values() if v != 'BENIGN']

# ── Chargement ─────────────────────────────────────────────────
print("Chargement données et modèles...", flush=True)
scaler    = joblib.load(MODEL_DIR / 'scaler_nfstream.joblib')
le        = joblib.load(DATA_DIR / 'processed' / 'label_encoder_nfstream.joblib')
xgb_multi = joblib.load(MODEL_DIR / 'xgb_multiclass_v2.joblib')

df = pd.read_csv(DATA_DIR / 'cicids2017_nfstream_labeled.csv', low_memory=False)
df = df[df['Label'] != 'UNLABELED'].copy()

X_raw = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
X     = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_NAMES)
y     = df['Label'].values

print(f"Flows: {len(X):,} | Features: {len(FEATURE_NAMES)}", flush=True)

# ── SHAP Explainer ─────────────────────────────────────────────
# Échantillon stratifié pour SHAP (2000 flows max pour performance)
np.random.seed(42)
sample_idx = []
for label in np.unique(y):
    idx = np.where(y == label)[0]
    n   = min(len(idx), 200)
    sample_idx.extend(np.random.choice(idx, n, replace=False))
sample_idx = np.array(sample_idx)

X_sample = X.iloc[sample_idx]
y_sample = y[sample_idx]

print(f"Échantillon SHAP: {len(X_sample)} flows", flush=True)
print("Calcul SHAP values (TreeExplainer)...", flush=True)

explainer   = shap.TreeExplainer(xgb_multi)
shap_values = explainer.shap_values(X_sample)
# shap_values : liste de 9 arrays (un par classe)
print(f"SHAP values shape: {np.array(shap_values).shape}", flush=True)

# ── 1. Global feature importance (mean |SHAP|) ─────────────────
print("\n[1/4] Global feature importance...", flush=True)
# shap_values shape : (n_samples, n_features, n_classes) ou liste
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # Nouvelle API SHAP : (n_samples, n_features, n_classes)
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    sv_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
else:
    # Ancienne API : liste de n_classes arrays (n_samples, n_features)
    mean_abs_shap = np.mean([np.abs(shap_values[i]) for i in range(len(LABEL_MAP))], axis=(0, 1))
    sv_per_class = shap_values

feat_importance = pd.DataFrame({
    'feature': FEATURE_NAMES,
    'mean_abs_shap': mean_abs_shap.tolist()
}).sort_values('mean_abs_shap', ascending=False)

feat_importance.to_csv(RESULTS_DIR / 'shap_global_importance.csv', index=False)
print(f"Top 10 features globales:")
print(feat_importance.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top20 = feat_importance.head(20)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20)))[::-1]
bars = ax.barh(range(len(top20)), top20['mean_abs_shap'], color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('Global Feature Importance — XGBoost Multiclass\n(IDS-KMUTT v2, NFStream features)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_global_importance.png", flush=True)

# ── 2. Per-class top features ──────────────────────────────────
print("\n[2/4] Per-class SHAP top features...", flush=True)
class_top_features = {}
for class_id, class_name in LABEL_MAP.items():
    sv = np.abs(shap_values[class_id])
    mean_sv = sv.mean(axis=0)
    top_idx = np.argsort(mean_sv)[::-1][:10]
    class_top_features[class_name] = [
        (FEATURE_NAMES[i], float(mean_sv[i])) for i in top_idx
    ]

# Heatmap top features par classe
top_features_global = feat_importance['feature'].head(20).tolist()
heatmap_data = np.zeros((len(LABEL_MAP), len(top_features_global)))
for class_id, class_name in LABEL_MAP.items():
    sv = np.abs(sv_per_class[class_id]).mean(axis=0)
    for j, feat in enumerate(top_features_global):
        feat_idx = FEATURE_NAMES.index(feat)
        heatmap_data[class_id, j] = sv[feat_idx]

# Normaliser par ligne pour voir les patterns relatifs
row_max = heatmap_data.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
heatmap_norm = heatmap_data / row_max

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(heatmap_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(top_features_global)))
ax.set_xticklabels(top_features_global, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(LABEL_MAP)))
ax.set_yticklabels([LABEL_MAP[i] for i in range(len(LABEL_MAP))], fontsize=10)
ax.set_title('SHAP Feature Importance per Class — XGBoost Multiclass\n(normalized per class)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized mean |SHAP|')
# Annotations
for i in range(len(LABEL_MAP)):
    for j in range(len(top_features_global)):
        val = heatmap_norm[i, j]
        if val > 0.5:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_perclass_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_perclass_heatmap.png", flush=True)

# ── 3. Summary plot par classe (top 4 attaques) ────────────────
print("\n[3/4] SHAP summary plots...", flush=True)
key_classes = ['PortScan', 'DDoS', 'SSH-Patator', 'DoS']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHAP Summary — Top Attack Classes\n(IDS-KMUTT v2 XGBoost Multiclass)', fontsize=13, fontweight='bold')

for ax, class_name in zip(axes.flatten(), key_classes):
    class_id  = [k for k, v in LABEL_MAP.items() if v == class_name][0]
    sv        = sv_per_class[class_id]
    mean_sv   = np.abs(sv).mean(axis=0)
    top_idx   = np.argsort(mean_sv)[::-1][:12]
    top_feats = [FEATURE_NAMES[i] for i in top_idx]
    top_vals  = mean_sv[top_idx]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_idx)))[::-1]
    ax.barh(range(len(top_idx)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.set_title(class_name, fontweight='bold', fontsize=11)
    ax.set_xlabel('Mean |SHAP value|', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_summary_key_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_summary_key_classes.png", flush=True)

# ── 4. Rapport texte ───────────────────────────────────────────
print("\n[4/4] Rapport SHAP...", flush=True)
report_lines = ["=" * 70,
                "SHAP ANALYSIS REPORT — IDS-KMUTT v2 (XGBoost Multiclass)",
                "=" * 70, ""]

report_lines.append("GLOBAL TOP 15 FEATURES (mean |SHAP| across all classes):")
for i, row in feat_importance.head(15).iterrows():
    report_lines.append(f"  {row['feature']:<40} {row['mean_abs_shap']:.6f}")

report_lines.append("")
report_lines.append("PER-CLASS TOP 5 FEATURES:")
for class_name, feats in class_top_features.items():
    report_lines.append(f"\n  [{class_name}]")
    for feat, val in feats[:5]:
        report_lines.append(f"    {feat:<40} {val:.6f}")

report = "\n".join(report_lines)
print(report)
with open(RESULTS_DIR / 'shap_report.txt', 'w') as f:
    f.write(report)

print("\n✅ Analyse SHAP terminée. Fichiers dans results/:")
for f in sorted(RESULTS_DIR.glob('shap_*')):
    print(f"  {f.name}")
