"""Isotonic recalibration for the v2 CNN on Zenodo hold-out (30/70 split, 5 seeds)."""
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
from metrics import full_metrics, calibration_diag

ART = Path("analysis/artifacts")
OUT = ART / "cnn_v2_recal"
OUT.mkdir(exist_ok=True, parents=True)
FIGS = Path("analysis/figs")


def op_points(y_true, y_prob):
    thr = np.round(np.arange(0.01, 1.00, 0.01), 2)
    best_f1, best_j = -1, -1
    r_f1 = r_j = None
    for t in thr:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        j = sens + spec - 1
        if f1 > best_f1:
            best_f1 = f1; r_f1 = {"threshold": float(t), "sensitivity": float(sens),
                                  "specificity": float(spec), "f1": float(f1),
                                  "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
        if j > best_j:
            best_j = j; r_j = {"threshold": float(t), "sensitivity": float(sens),
                               "specificity": float(spec), "f1": float(f1),
                               "youden_j": float(j),
                               "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    return r_f1, r_j


def main():
    y_true_all = np.load(ART / "cnn_v2" / "test_labels.npy")
    y_prob_all = np.load(ART / "cnn_v2" / "test_probs.npy")
    print(f"External Zenodo n={len(y_true_all)}  prevalence={y_true_all.mean():.3f}")

    rows = []
    for seed in [42, 43, 44, 45, 46]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y_true_all)); rng.shuffle(idx)
        n_cal = int(0.30 * len(idx))
        ci, ei = idx[:n_cal], idx[n_cal:]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(y_prob_all[ci], y_true_all[ci])
        y_prob_cal = iso.transform(y_prob_all[ei])
        y_true_ev = y_true_all[ei]; y_raw_ev = y_prob_all[ei]

        raw_m = full_metrics(y_true_ev, y_raw_ev)
        cal_m = full_metrics(y_true_ev, y_prob_cal)
        raw_c = calibration_diag(y_true_ev, y_raw_ev, n_bins=10)
        cal_c = calibration_diag(y_true_ev, y_prob_cal, n_bins=10)
        raw_f1, raw_j = op_points(y_true_ev, y_raw_ev)
        cal_f1, cal_j = op_points(y_true_ev, y_prob_cal)

        rows.append({"seed": seed, "n_cal": int(n_cal), "n_eval": int(len(ei)),
                     "raw": {"metrics": raw_m, "calibration": raw_c,
                             "op_max_f1": raw_f1, "op_youden": raw_j},
                     "calibrated": {"metrics": cal_m, "calibration": cal_c,
                                    "op_max_f1": cal_f1, "op_youden": cal_j}})

    def agg(section, subkey, metric_key):
        vals = np.array([r[section][subkey][metric_key] for r in rows])
        return vals.mean(), vals.std()

    print("\n5-seed aggregate (mean ± SD)")
    print("\nDiscrimination (unchanged by isotonic — order preserving):")
    for m in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        raw_m, raw_s = agg("raw", "metrics", m)
        cal_m, cal_s = agg("calibrated", "metrics", m)
        print(f"  {m:15s}  raw={raw_m:.3f} ± {raw_s:.3f}   cal={cal_m:.3f} ± {cal_s:.3f}")

    print("\nCalibration diagnostics:")
    for c in ["brier", "ece", "slope"]:
        rm, rs = agg("raw", "calibration", c)
        cm, cs = agg("calibrated", "calibration", c)
        print(f"  {c:10s}  raw={rm:.3f} ± {rs:.3f}   cal={cm:.3f} ± {cs:.3f}")

    print("\nMax-F1 operating point (calibrated):")
    for k in ["threshold", "sensitivity", "specificity", "f1", "fp", "fn"]:
        vals = np.array([r["calibrated"]["op_max_f1"][k] for r in rows])
        print(f"  {k:12s}  mean={vals.mean():.3f}  SD={vals.std():.3f}")

    print("\nYouden-J operating point (calibrated):")
    for k in ["threshold", "sensitivity", "specificity", "youden_j", "fp", "fn"]:
        vals = np.array([r["calibrated"]["op_youden"][k] for r in rows])
        print(f"  {k:12s}  mean={vals.mean():.3f}  SD={vals.std():.3f}")

    with open(OUT / "recal.json", "w") as f:
        json.dump({"seeds": [42, 43, 44, 45, 46], "runs": rows}, f, indent=2, default=float)
    print(f"\n→ {OUT}/recal.json")

    # Plot
    r0 = rows[0]
    rng = np.random.default_rng(42)
    idx = np.arange(len(y_true_all)); rng.shuffle(idx)
    n_cal = int(0.30 * len(idx))
    ci, ei = idx[:n_cal], idx[n_cal:]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(y_prob_all[ci], y_true_all[ci])
    y_ev = y_true_all[ei]
    y_raw = y_prob_all[ei]
    y_cal = iso.transform(y_raw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0, 1, 11)
    for ax, y_prob, title, color in [
        (axes[0], y_raw, "v2 CNN raw", "#c0392b"),
        (axes[1], y_cal, "v2 CNN + isotonic on Zenodo", "#2a9d8f"),
    ]:
        ids = np.digitize(y_prob, bins[1:-1])
        xs, ys = [], []
        for b in range(10):
            m = ids == b
            if m.sum() == 0: continue
            xs.append(y_prob[m].mean()); ys.append(y_ev[m].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(xs, ys, "o-", color=color, label="Model")
        ax.set_xlabel("Predicted p"); ax.set_ylabel("Observed +")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    axes[2].hist(y_raw, bins=30, alpha=0.5, label="raw", color="#c0392b")
    axes[2].hist(y_cal, bins=30, alpha=0.5, label="calibrated", color="#2a9d8f")
    axes[2].set_xlabel("Predicted probability"); axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.suptitle(f"v2 CNN — Zenodo isotonic recalibration (n_cal={n_cal}, n_eval={len(ei)})")
    plt.tight_layout()
    plt.savefig(FIGS / "cnn_v2_recal.png", dpi=140)
    plt.close()
    print(f"→ {FIGS}/cnn_v2_recal.png")


if __name__ == "__main__":
    main()
