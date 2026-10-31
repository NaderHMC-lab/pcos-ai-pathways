"""Nested 5-outer × 5-inner CV for tabular winner selection.

Removes the mild test-peeking bias from picking winner by held-out AUC.
For each outer fold:
  1. Inner 5-fold CV tunes XGBoost (top model from earlier) via Optuna 20 trials.
  2. Retrain on the outer-train, evaluate on the outer-test.
Outer-test AUCs give an honest generalisation estimate.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import data as data_mod
from config import RANDOM_STATE, CATEGORICAL
from metrics import full_metrics, bootstrap_ci
from models import _obj_xgb, _pipe, make_preprocessor_tree
from xgboost import XGBClassifier

OUT = Path("analysis/artifacts/nested_cv")
OUT.mkdir(exist_ok=True, parents=True)


def main():
    df = data_mod.load_all()
    feature_cols = data_mod.get_feature_columns(df)
    num_cols = [c for c in feature_cols if c not in CATEGORICAL]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL]

    X = df[feature_cols].copy()
    y = df["pcos"].astype(int).values
    print(f"Nested-CV on n={len(y)}  positives={int(y.sum())}  ({y.mean():.3f})")

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer_metrics = []
    fold_probs, fold_true = [], []

    for k, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        t0 = time.time()
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + k)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + k))
        study.optimize(lambda t: _obj_xgb(t, X_tr, y_tr, num_cols, cat_cols, inner),
                       n_trials=20, show_progress_bar=False)
        best = study.best_params
        clf = XGBClassifier(**best, eval_metric="logloss",
                            random_state=RANDOM_STATE + k, tree_method="hist", n_jobs=-1)
        pipe = _pipe(make_preprocessor_tree(num_cols, cat_cols), clf)
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        m = full_metrics(y_te, y_prob)
        outer_metrics.append({"fold": k, "n_test": int(len(y_te)),
                              "inner_best_cv_auc": float(study.best_value),
                              **{k: float(v) for k, v in m.items()},
                              "elapsed_s": round(time.time() - t0, 1)})
        fold_probs.append(y_prob); fold_true.append(y_te)
        print(f"  fold {k}: inner-CV-AUC={study.best_value:.4f}  outer-test-AUC={m['roc_auc']:.4f}  "
              f"sens={m['sensitivity']:.3f}  ({time.time()-t0:.1f}s)")

    outer_df = pd.DataFrame(outer_metrics)
    # Aggregate: mean±SD across folds
    agg = {}
    for k in ["roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        arr = outer_df[k].values
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "min": float(arr.min()), "max": float(arr.max())}

    # Pooled bootstrap CI across all folds' predictions (concatenated)
    all_prob = np.concatenate(fold_probs)
    all_true = np.concatenate(fold_true)
    pooled_ci = bootstrap_ci(all_true, all_prob, n_iter=1000, seed=RANDOM_STATE)

    print(f"\n{'='*70}\nNested-CV outer folds — mean ± SD\n{'='*70}")
    for k, v in agg.items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}   "
              f"[range: {v['min']:.4f}, {v['max']:.4f}]")

    print(f"\nPooled bootstrap 95% CIs (n={len(all_true)}):")
    for k, v in pooled_ci.items():
        print(f"  {k:20s}  {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    with open(OUT / "nested_cv.json", "w") as f:
        json.dump({
            "outer_folds": outer_metrics,
            "aggregate": agg,
            "pooled_bootstrap_ci": pooled_ci,
            "n_pooled_predictions": int(len(all_true)),
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/nested_cv.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
