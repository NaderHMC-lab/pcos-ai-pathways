"""Paired-bootstrap ΔAUC test.

Two comparisons:
  (A) Tabular ML (calibrated XGBoost) vs 12-feature Clinical LR — same test rows
  (B) CNN internal (contaminated Kaggle test) vs external (truly-external Zenodo)
      — different rows, unpaired, so we use independent bootstrap

Reports:
  - Point-estimate AUCs and ΔAUC
  - 95% CI on ΔAUC (paired bootstrap for A, independent bootstrap for B)
  - Two-sided p-value proxy (fraction of bootstrap samples with sign(ΔAUC) reversed)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import data as data_mod
from config import CATEGORICAL, RANDOM_STATE

ART = Path("analysis/artifacts")
OUT = ART / "auc_diff"
OUT.mkdir(exist_ok=True, parents=True)


def paired_bootstrap_auc(y_true, y_prob_a, y_prob_b, n_iter=2000, seed=42):
    """Paired bootstrap ΔAUC = AUC(A) − AUC(B)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob_a = np.asarray(y_prob_a); y_prob_b = np.asarray(y_prob_b)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1); neg_idx = np.flatnonzero(y_true == 0)

    deltas = []
    for _ in range(n_iter):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        y = y_true[idx]
        if len(np.unique(y)) < 2:
            continue
        auc_a = roc_auc_score(y, y_prob_a[idx])
        auc_b = roc_auc_score(y, y_prob_b[idx])
        deltas.append(auc_a - auc_b)
    deltas = np.array(deltas)

    auc_a_full = roc_auc_score(y_true, y_prob_a)
    auc_b_full = roc_auc_score(y_true, y_prob_b)
    delta_hat = auc_a_full - auc_b_full
    p_two_sided = float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
    return {
        "auc_a": float(auc_a_full), "auc_b": float(auc_b_full),
        "delta_auc": float(delta_hat),
        "delta_ci": [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
        "p_two_sided": p_two_sided,
        "n_bootstrap": int(n_iter), "n_paired": int(len(y_true)),
    }


def unpaired_bootstrap_auc(y_a, p_a, y_b, p_b, n_iter=2000, seed=42):
    """Independent bootstrap ΔAUC when the two evaluations are on different rows."""
    y_a = np.asarray(y_a).astype(int); y_b = np.asarray(y_b).astype(int)
    rng = np.random.default_rng(seed)
    pos_a, neg_a = np.flatnonzero(y_a == 1), np.flatnonzero(y_a == 0)
    pos_b, neg_b = np.flatnonzero(y_b == 1), np.flatnonzero(y_b == 0)

    deltas = []
    for _ in range(n_iter):
        pa = rng.choice(pos_a, size=len(pos_a), replace=True)
        na = rng.choice(neg_a, size=len(neg_a), replace=True)
        pb = rng.choice(pos_b, size=len(pos_b), replace=True)
        nb = rng.choice(neg_b, size=len(neg_b), replace=True)
        ia, ib = np.concatenate([pa, na]), np.concatenate([pb, nb])
        auc_a = roc_auc_score(y_a[ia], p_a[ia])
        auc_b = roc_auc_score(y_b[ib], p_b[ib])
        deltas.append(auc_a - auc_b)
    deltas = np.array(deltas)
    auc_a_full = roc_auc_score(y_a, p_a)
    auc_b_full = roc_auc_score(y_b, p_b)
    delta_hat = auc_a_full - auc_b_full
    p_two_sided = float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
    return {
        "auc_a": float(auc_a_full), "auc_b": float(auc_b_full),
        "delta_auc": float(delta_hat),
        "delta_ci": [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
        "p_two_sided": p_two_sided,
        "n_bootstrap": int(n_iter),
        "n_a": int(len(y_a)), "n_b": int(len(y_b)),
    }


def clinical_lr_features():
    return ["age_yrs", "bmi", "follicle_no_l", "follicle_no_r", "amh_ng_ml",
            "lh_miu_ml", "fsh_miu_ml", "hair_growth_y_n", "pimples_y_n",
            "skin_darkening_y_n", "cycle_r_i", "waist_hip_ratio"]


def main():
    print("=" * 70)
    print("Comparison A: Tabular ML vs 12-feature Clinical LR")
    print("=" * 70)
    df = data_mod.load_all()
    feats = data_mod.get_feature_columns(df)
    X = df[feats].copy(); y = df["pcos"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y,
                                              random_state=RANDOM_STATE)
    # ML probs
    ml = joblib.load(ART / "pcos_best" / "calibrated_model.pkl")
    ml_prob = ml.predict_proba(X_te)[:, 1]

    # Clinical LR probs — retrained on same X_tr
    feats_lr = clinical_lr_features()
    pre = SKPipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())])
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_pipe = SKPipeline([("pre", pre), ("clf", lr)])
    lr_cal = CalibratedClassifierCV(lr_pipe, method="isotonic", cv=5)
    lr_cal.fit(X_tr[feats_lr], y_tr)
    lr_prob = lr_cal.predict_proba(X_te[feats_lr])[:, 1]

    A = paired_bootstrap_auc(y_te, ml_prob, lr_prob, n_iter=2000, seed=42)
    print(f"  ML   AUC  = {A['auc_a']:.4f}")
    print(f"  LR12 AUC  = {A['auc_b']:.4f}")
    print(f"  ΔAUC (ML − LR)  = {A['delta_auc']:+.4f}   "
          f"95% CI = [{A['delta_ci'][0]:+.4f}, {A['delta_ci'][1]:+.4f}]")
    print(f"  p (two-sided)   = {A['p_two_sided']:.3f}   "
          f"(n_paired={A['n_paired']})")

    print("\n" + "=" * 70)
    print("Comparison B: CNN internal (Kaggle test) vs external (truly-external Zenodo)")
    print("=" * 70)
    # Internal
    y_int = np.load(ART / "cnn_ultrasound" / "test_labels.npy")
    p_int = np.load(ART / "cnn_ultrasound" / "test_probs.npy")
    # External clean (removing Kaggle overlap)
    ext = pd.read_csv(ART / "external_val" / "preds_y_pco.csv")
    overlap = set(json.load(open(ART / "dedup" / "zenodo_overlap_names.json")))
    ext_clean = ext[~ext["imagePath"].isin(overlap)]
    y_ext = ext_clean["y_true"].values
    p_ext = ext_clean["y_prob"].values

    B = unpaired_bootstrap_auc(y_int, p_int, y_ext, p_ext, n_iter=2000, seed=42)
    print(f"  Internal  AUC = {B['auc_a']:.4f}  (n={B['n_a']})")
    print(f"  External  AUC = {B['auc_b']:.4f}  (n={B['n_b']})")
    print(f"  ΔAUC (int − ext) = {B['delta_auc']:+.4f}   "
          f"95% CI = [{B['delta_ci'][0]:+.4f}, {B['delta_ci'][1]:+.4f}]")
    print(f"  p (two-sided)    = {B['p_two_sided']:.3f}")

    with open(OUT / "auc_diff.json", "w") as f:
        json.dump({"tabular_ml_vs_clinical_lr": A,
                   "cnn_internal_vs_external": B},
                  f, indent=2, default=float)
    print(f"\n→ {OUT}/auc_diff.json")


if __name__ == "__main__":
    main()
