"""Metrics: bootstrap CIs, calibration diagnostics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix,
)


def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def full_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    sens, spec = sens_spec(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }


def bootstrap_ci(y_true, y_prob, n_iter=1000, threshold=0.5, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    keys = list(full_metrics(y_true, y_prob, threshold).keys())
    samples = {k: [] for k in keys}
    for _ in range(n_iter):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) else np.array([], dtype=int)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) else np.array([], dtype=int)
        idx = np.concatenate([p, n])
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = full_metrics(y_true[idx], y_prob[idx], threshold)
        for k, v in m.items():
            if v is not None and not np.isnan(v):
                samples[k].append(v)
    out = {}
    for k in keys:
        arr = np.asarray(samples[k])
        out[k] = {
            "mean": float(np.nanmean(arr)) if len(arr) else np.nan,
            "ci_low": float(np.nanpercentile(arr, 2.5)) if len(arr) else np.nan,
            "ci_high": float(np.nanpercentile(arr, 97.5)) if len(arr) else np.nan,
        }
    return out


def calibration_diag(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y_true, y_prob)
    p_bar = y_true.mean()
    no_skill = p_bar * (1 - p_bar)
    brier_skill = 1 - brier / no_skill if no_skill > 0 else np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0; mce = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0: continue
        conf = y_prob[mask].mean(); acc = y_true[mask].mean(); gap = abs(conf - acc)
        ece += mask.sum() / len(y_prob) * gap
        mce = max(mce, gap)
    logit_p = np.log(y_prob / (1 - y_prob))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit_p.reshape(-1, 1), y_true)
    return {
        "brier": float(brier), "no_skill_brier": float(no_skill),
        "brier_skill": float(brier_skill), "ece": float(ece), "mce": float(mce),
        "slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]),
    }
