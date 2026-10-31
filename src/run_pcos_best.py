"""Best-effort tabular model: same pipeline as run_pcos.py but with 60 Optuna trials.

Saves the calibrated winner + full metadata under artifacts/pcos_best/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

import data as data_mod
from config import RANDOM_STATE, BOOT_ITERS, CATEGORICAL
from metrics import full_metrics, bootstrap_ci, calibration_diag
from models import base_models, tune

ART = Path("analysis/artifacts")


def run():
    print("="*70, "\n  TABULAR — best-effort run (60 Optuna trials)\n", "="*70)

    df = data_mod.load_all()
    feature_cols = data_mod.get_feature_columns(df)
    num_cols = [c for c in feature_cols if c not in CATEGORICAL]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL]

    X = df[feature_cols].copy()
    y = df["pcos"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"features={len(feature_cols)}  train={len(y_tr)}  test={len(y_te)}")

    # 8-model zoo
    print("\n[1/3] Baseline zoo:")
    zoo = base_models(num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    zoo_rows = []
    for name, pipe in zoo.items():
        t0 = time.time()
        try:
            cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        except Exception as e:
            print(f"  [{name}] failed: {e}"); continue
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        test = full_metrics(y_te, y_prob)
        zoo_rows.append({"model": name, "cv_auc": float(cv_auc),
                         "test_auc": test["roc_auc"], "test_sens": test["sensitivity"],
                         "test_f1": test["f1"]})
        print(f"  {name:12s}  CV={cv_auc:.4f}  Test={test['roc_auc']:.4f}  "
              f"Sens={test['sensitivity']:.3f}  F1={test['f1']:.3f}  ({time.time()-t0:.1f}s)")

    # Tune top-3 tree models with 60 Optuna trials each
    print("\n[2/3] Optuna tuning (60 trials each):")
    tuned = {}
    for name in ["xgboost", "lightgbm", "catboost"]:
        t0 = time.time()
        pipe, best_params, best_cv = tune(name, X_tr, y_tr, num_cols, cat_cols, n_trials=60)
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        test = full_metrics(y_te, y_prob)
        tuned[name] = {"pipe": pipe, "best_params": best_params,
                       "cv_auc": float(best_cv),
                       "test_auc": test["roc_auc"], "test_sens": test["sensitivity"],
                       "test_f1": test["f1"]}
        print(f"  [{name}]  CV={best_cv:.4f}  Test={test['roc_auc']:.4f}  "
              f"Sens={test['sensitivity']:.3f}  ({time.time()-t0:.1f}s)")

    # Winner: max held-out AUC, tie-break on sensitivity, then on CV AUC
    candidates = {**{n: (r["test_auc"], r["test_sens"], r["cv_auc"], None)
                     for n, r in zip([r["model"] for r in zoo_rows], zoo_rows) if n not in tuned},
                  **{n: (r["test_auc"], r["test_sens"], r["cv_auc"], r["pipe"])
                     for n, r in tuned.items()}}
    # Also add the tuned tree pipes so a tuned version can win over an untuned one
    winner_name = max(candidates, key=lambda k: (candidates[k][0], candidates[k][1], candidates[k][2]))
    winner_pipe = candidates[winner_name][3]
    if winner_pipe is None:
        # Untuned winner — refit
        winner_pipe = zoo[winner_name]; winner_pipe.fit(X_tr, y_tr)
    print(f"\nWinner: {winner_name}  "
          f"(Test-AUC={candidates[winner_name][0]:.4f}, Sens={candidates[winner_name][1]:.3f})")

    # Isotonic calibration + bootstrap CI
    print("\n[3/3] Isotonic calibration + bootstrap CI:")
    calibrated = CalibratedClassifierCV(winner_pipe, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    y_prob = calibrated.predict_proba(X_te)[:, 1]

    headline = full_metrics(y_te, y_prob)
    ci = bootstrap_ci(y_te, y_prob, n_iter=BOOT_ITERS, seed=RANDOM_STATE)
    calib = calibration_diag(y_te, y_prob, n_bins=10)

    for k, v in ci.items():
        print(f"  {k:20s}  {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print(f"  Calibration: Brier={calib['brier']:.4f}  ECE={calib['ece']:.4f}  slope={calib['slope']:.3f}")

    # Save
    out = ART / "pcos_best"; out.mkdir(exist_ok=True, parents=True)
    joblib.dump(calibrated, out / "calibrated_model.pkl")
    joblib.dump({"winner": winner_name, "features": feature_cols,
                 "num_cols": num_cols, "cat_cols": cat_cols,
                 "test_idx": X_te.index.tolist()},
                out / "meta.pkl")
    with open(out / "results.json", "w") as f:
        json.dump({
            "winner": winner_name,
            "n_features": len(feature_cols),
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "headline": headline,
            "bootstrap_ci": ci,
            "calibration": calib,
            "zoo": zoo_rows,
            "tuned_best_params": {n: r["best_params"] for n, r in tuned.items()},
        }, f, indent=2, default=float)
    pd.DataFrame({"y_true": y_te, "y_prob": y_prob}, index=X_te.index).to_csv(
        out / "test_predictions.csv")
    print(f"\n→ {out}/")


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
