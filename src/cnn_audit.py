"""CNN audit stage: bootstrap 95% CIs + Zenodo threshold recalibration.

Requires that train_cnn.py and external_val.py have already been run
(reads their saved predictions).
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import bootstrap_ci

ART = Path("analysis/artifacts")
FIGS = Path("analysis/figs")
FIGS.mkdir(exist_ok=True, parents=True)


def bootstrap_and_save(name, y_true, y_prob, out_json):
    ci = bootstrap_ci(y_true, y_prob, n_iter=1000, seed=42)
    print(f"\n[{name}]  n={len(y_true)}  prevalence={y_true.mean():.3f}")
    for k, v in ci.items():
        print(f"  {k:20s}  {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    with open(out_json, "w") as f:
        json.dump(ci, f, indent=2)
    return ci


def threshold_recalibration(y_true, y_prob, out_json, out_fig):
    """Sweep 0.01→0.99, find operating points, plot metrics vs threshold."""
    from sklearn.metrics import (
        f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    )
    thr = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = []
    for t in thr:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        j = sens + spec - 1  # Youden
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "youden_j": float(j),
        })
    df = pd.DataFrame(rows)

    default_t = 0.50
    balanced_t = float(df.loc[df["f1"].idxmax(), "threshold"])
    youden_t = float(df.loc[df["youden_j"].idxmax(), "threshold"])
    screen_mask = df["sensitivity"] >= 0.95
    screen_t = float(df[screen_mask]["threshold"].min()) if screen_mask.any() else float("nan")
    conf_mask = df["specificity"] >= 0.95
    conf_t = float(df[conf_mask]["threshold"].max()) if conf_mask.any() else float("nan")

    ops = {
        "default": default_t,
        "balanced_max_f1": balanced_t,
        "youden_j": youden_t,
        "screening_sens_95": screen_t,
        "confirmation_spec_95": conf_t,
    }

    def row_at(t):
        return df.iloc[(df["threshold"] - t).abs().argsort().iloc[0]].to_dict()

    op_metrics = {name: row_at(t) for name, t in ops.items() if not np.isnan(t)}

    print(f"\nOperating points on Zenodo (PCO-visible):")
    for name, m in op_metrics.items():
        print(f"  {name:22s} thr={m['threshold']:.2f}  "
              f"acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}  F1={m['f1']:.3f}  J={m['youden_j']:+.3f}")

    with open(out_json, "w") as f:
        json.dump({
            "auc": float(roc_auc_score(y_true, y_prob)),
            "operating_points": ops,
            "operating_metrics": op_metrics,
            "sweep": df.to_dict(orient="records"),
        }, f, indent=2)

    # Plot: metrics vs threshold, with vertical lines at each operating point
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, color in [("sensitivity", "#e63946"), ("specificity", "#457b9d"),
                       ("f1", "#2a9d8f"), ("youden_j", "#8e44ad")]:
        ax.plot(df["threshold"], df[col], "-", label=col, color=color, linewidth=1.5)
    ax.axvline(default_t, ls="--", color="black", alpha=0.5, label=f"default (0.50)")
    ax.axvline(youden_t, ls=":", color="#8e44ad", linewidth=2, label=f"Youden J ({youden_t:.2f})")
    ax.axvline(balanced_t, ls=":", color="#2a9d8f", linewidth=2, label=f"balanced F1 ({balanced_t:.2f})")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value")
    ax.set_title("Zenodo threshold recalibration — CNN")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(out_fig, dpi=140); plt.close()
    print(f"→ {out_fig.name}")

    return op_metrics


def main():
    OUT = ART / "cnn_audit"
    OUT.mkdir(exist_ok=True)

    # --- Load internal (Kaggle) predictions ---
    y_true_int = np.load(ART / "cnn_ultrasound" / "test_labels.npy")
    y_prob_int = np.load(ART / "cnn_ultrasound" / "test_probs.npy")

    ci_int = bootstrap_and_save("Internal (anaghachoudhari test)",
                                y_true_int, y_prob_int,
                                OUT / "ci_internal.json")

    # --- Load external (Zenodo) predictions ---
    ext_pco = pd.read_csv(ART / "external_val" / "preds_y_pco.csv")
    y_true_pco = ext_pco["y_true"].values
    y_prob_pco = ext_pco["y_prob"].values
    ci_pco = bootstrap_and_save("External Zenodo (PCO-visible)",
                                y_true_pco, y_prob_pco,
                                OUT / "ci_external_pco.json")

    ext_ab = pd.read_csv(ART / "external_val" / "preds_y_abnormal.csv")
    y_true_ab = ext_ab["y_true"].values
    y_prob_ab = ext_ab["y_prob"].values
    ci_ab = bootstrap_and_save("External Zenodo (abnormal)",
                                y_true_ab, y_prob_ab,
                                OUT / "ci_external_abnormal.json")

    # --- Threshold recalibration on Zenodo (PCO-visible = closer to training label) ---
    print("\n" + "=" * 70)
    print("Threshold recalibration on Zenodo (PCO-visible)")
    print("=" * 70)
    op_metrics = threshold_recalibration(y_true_pco, y_prob_pco,
                                         OUT / "threshold_recal.json",
                                         FIGS / "cnn_threshold_zenodo.png")

    # Combined summary
    summary = {
        "internal_ci": ci_int,
        "external_pco_ci": ci_pco,
        "external_abnormal_ci": ci_ab,
        "zenodo_operating_points": op_metrics,
    }
    with open(OUT / "cnn_audit.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n→ {OUT}/cnn_audit.json")


if __name__ == "__main__":
    main()
