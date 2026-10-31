"""Multi-seed tabular variance analysis (5 seeds)."""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import data as data_mod
from config import CATEGORICAL
from metrics import full_metrics
from models import _pipe, make_preprocessor_tree
from xgboost import XGBClassifier

ART = Path("analysis/artifacts")
OUT = ART / "multiseed"
OUT.mkdir(exist_ok=True, parents=True)
SEEDS = [42, 43, 44, 45, 46]


def main():
    params = json.load(open(ART / "pcos_best" / "results.json"))["tuned_best_params"]["xgboost"]
    df = data_mod.load_all()
    feats = data_mod.get_feature_columns(df)
    num = [c for c in feats if c not in CATEGORICAL]
    cat = [c for c in feats if c in CATEGORICAL]
    X = df[feats]; y = df["pcos"].astype(int).values

    rows = []
    for s in SEEDS:
        t0 = time.time()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
        clf = XGBClassifier(**params, eval_metric="logloss",
                            random_state=s, tree_method="hist", n_jobs=-1)
        pipe = _pipe(make_preprocessor_tree(num, cat), clf)
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
        cal.fit(X_tr, y_tr)
        y_prob = cal.predict_proba(X_te)[:, 1]
        m = full_metrics(y_te, y_prob)
        rows.append({"seed": s, "n_test": int(len(y_te)),
                     **{k: float(v) for k, v in m.items()}})
        print(f"  seed={s}  AUC={m['roc_auc']:.4f}  Sens={m['sensitivity']:.3f}  "
              f"Spec={m['specificity']:.3f}  ({time.time()-t0:.1f}s)")

    keys = ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]
    agg = {}
    for k in keys:
        arr = np.array([r[k] for r in rows])
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "min": float(arr.min()), "max": float(arr.max()),
                  "values": arr.tolist()}
    print("\nAggregate (mean ± SD):")
    for k, v in agg.items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}   "
              f"[{v['min']:.4f}, {v['max']:.4f}]")
    with open(OUT / "multiseed_tabular.json", "w") as f:
        json.dump({"seeds": SEEDS, "rows": rows, "aggregate": agg}, f, indent=2, default=float)
    print(f"\n→ {OUT}/multiseed_tabular.json")


if __name__ == "__main__":
    main()
