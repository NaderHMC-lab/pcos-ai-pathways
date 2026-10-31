"""Fit isotonic calibrator on Zenodo hold-out → evaluate on eval split.

Uses only the truly-external Zenodo subset (n=1304 after removing Kaggle dupes).
Repeats over 5 seeds for variance.
"""
import json
import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix, f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from metrics import calibration_diag, full_metrics

ART = Path("analysis/artifacts")
FIGS = Path("analysis/figs")
OUT = ART / "isotonic_recal"
OUT.mkdir(exist_ok=True)


def op_points(y_true, y_prob):
    """Return default (0.5), max-F1, and Youden operating points."""
    thr = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = []
    for t in thr:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append({"threshold": float(t), "sensitivity": sens, "specificity": spec,
                     "f1": f1_score(y_true, y_pred, zero_division=0),
                     "youden_j": sens + spec - 1})
    df = pd.DataFrame(rows)
    m_f1 = df.loc[df["f1"].idxmax()].to_dict()
    m_j = df.loc[df["youden_j"].idxmax()].to_dict()
    default_idx = (df["threshold"] - 0.5).abs().idxmin()
    m_d = df.loc[default_idx].to_dict()
    return {"default_0.5": m_d, "max_f1": m_f1, "max_youden_j": m_j}


def run_one_seed(seed, clean_df):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clean_df))
    rng.shuffle(idx)
    n_cal = int(0.30 * len(idx))
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]
    cal = clean_df.iloc[cal_idx]
    ev = clean_df.iloc[eval_idx]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(cal["y_prob"].values, cal["y_true"].values)
    ev_prob_cal = iso.transform(ev["y_prob"].values)

    y_true = ev["y_true"].values
    y_prob_raw = ev["y_prob"].values

    # Uncalibrated eval-split metrics
    raw_m = full_metrics(y_true, y_prob_raw)
    raw_c = calibration_diag(y_true, y_prob_raw, n_bins=10)
    raw_ops = op_points(y_true, y_prob_raw)

    # Calibrated eval-split metrics
    cal_m = full_metrics(y_true, ev_prob_cal)
    cal_c = calibration_diag(y_true, ev_prob_cal, n_bins=10)
    cal_ops = op_points(y_true, ev_prob_cal)

    return {
        "seed": seed,
        "n_cal": int(n_cal), "n_eval": int(len(ev)),
        "prevalence_eval": float(y_true.mean()),
        "raw": {"metrics": raw_m, "calibration": raw_c, "operating_points": raw_ops},
        "calibrated": {"metrics": cal_m, "calibration": cal_c, "operating_points": cal_ops},
    }


def main():
    ext_pco = pd.read_csv(ART / "external_val" / "preds_y_pco.csv")
    overlap = set(json.load(open(ART / "dedup" / "zenodo_overlap_names.json")))
    clean = ext_pco[~ext_pco["imagePath"].isin(overlap)].reset_index(drop=True)
    print(f"Truly-external Zenodo n={len(clean)}  prevalence={clean['y_true'].mean():.3f}")

    # 5 seeds
    all_runs = [run_one_seed(s, clean) for s in [42, 43, 44, 45, 46]]

    # Aggregate mean/std across seeds
    def agg(field):
        arr = np.array([r[field[0]][field[1]][field[2]] for r in all_runs])
        return float(arr.mean()), float(arr.std())

    def compare(metric_key):
        raw_mean, raw_sd = agg(("raw", "metrics", metric_key))
        cal_mean, cal_sd = agg(("calibrated", "metrics", metric_key))
        return {"raw": (raw_mean, raw_sd), "calibrated": (cal_mean, cal_sd)}

    def compare_cal(cal_key):
        raw_mean, raw_sd = agg(("raw", "calibration", cal_key))
        cal_mean, cal_sd = agg(("calibrated", "calibration", cal_key))
        return {"raw": (raw_mean, raw_sd), "calibrated": (cal_mean, cal_sd)}

    print(f"\n{'='*70}\nAcross 5 seeds — mean ± SD:\n{'='*70}")
    print(f"\n Discrimination (should be unchanged by isotonic — order-preserving):")
    for m in ["accuracy", "roc_auc", "f1", "sensitivity", "specificity", "mcc"]:
        c = compare(m)
        print(f"  {m:15s}  raw={c['raw'][0]:.3f}±{c['raw'][1]:.3f}   "
              f"cal={c['calibrated'][0]:.3f}±{c['calibrated'][1]:.3f}")

    print(f"\n Calibration diagnostics:")
    for k in ["brier", "ece", "mce", "slope"]:
        c = compare_cal(k)
        print(f"  {k:10s}  raw={c['raw'][0]:.3f}±{c['raw'][1]:.3f}   "
              f"cal={c['calibrated'][0]:.3f}±{c['calibrated'][1]:.3f}")

    print(f"\n Operating point — max Youden J:")
    for prob_type in ["raw", "calibrated"]:
        thrs = np.array([r[prob_type]["operating_points"]["max_youden_j"]["threshold"] for r in all_runs])
        sens = np.array([r[prob_type]["operating_points"]["max_youden_j"]["sensitivity"] for r in all_runs])
        spec = np.array([r[prob_type]["operating_points"]["max_youden_j"]["specificity"] for r in all_runs])
        j = sens + spec - 1
        print(f"  {prob_type:12s}  thr={thrs.mean():.2f}±{thrs.std():.2f}   "
              f"sens={sens.mean():.3f}   spec={spec.mean():.3f}   J={j.mean():.3f}±{j.std():.3f}")

    # Save all runs
    with open(OUT / "isotonic_recal.json", "w") as f:
        json.dump({
            "n_zenodo_clean": int(len(clean)),
            "seeds": [42, 43, 44, 45, 46],
            "runs": all_runs,
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/isotonic_recal.json")

    # ------- Visualize: raw vs calibrated reliability (seed 42) -------
    r0 = all_runs[0]
    rng = np.random.default_rng(42)
    idx = np.arange(len(clean)); rng.shuffle(idx)
    n_cal = int(0.30 * len(idx))
    ev = clean.iloc[idx[n_cal:]]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(clean.iloc[idx[:n_cal]]["y_prob"], clean.iloc[idx[:n_cal]]["y_true"])
    y_true = ev["y_true"].values
    y_raw = ev["y_prob"].values
    y_cal = iso.transform(y_raw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: raw reliability
    bins = np.linspace(0, 1, 11)
    for ax, y_prob, title, color in [
        (axes[0], y_raw, "Before isotonic recalibration", "#c0392b"),
        (axes[1], y_cal, "After isotonic recalibration", "#2a9d8f"),
    ]:
        ids = np.digitize(y_prob, bins[1:-1])
        xs, ys, cts = [], [], []
        for b in range(10):
            m = ids == b
            if m.sum() == 0: continue
            xs.append(y_prob[m].mean()); ys.append(y_true[m].mean()); cts.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(xs, ys, "o-", color=color, label="Model")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive rate")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)

    # Panel 3: histogram of raw vs calibrated probs
    axes[2].hist(y_raw, bins=30, alpha=0.5, label="raw", color="#c0392b")
    axes[2].hist(y_cal, bins=30, alpha=0.5, label="calibrated", color="#2a9d8f")
    axes[2].set_xlabel("Predicted probability"); axes[2].set_ylabel("Count")
    axes[2].set_title("Probability distribution"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle(f"Isotonic recalibration on truly-external Zenodo (n_cal={n_cal}, n_eval={len(ev)}, seed=42)")
    plt.tight_layout()
    plt.savefig(FIGS / "isotonic_recalibration.png", dpi=140)
    plt.close()
    print(f"→ {FIGS}/isotonic_recalibration.png")


if __name__ == "__main__":
    main()
