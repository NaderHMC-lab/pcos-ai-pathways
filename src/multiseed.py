"""Multi-seed variance analysis for both tabular and CNN arms.

Tabular: 5 seeds × (train/test split + XGBoost training + calibration) → mean±SD.
CNN: 5 seeds × (train/val split + MobileNetV3 head-only 3 epochs on frozen backbone).
     Full fine-tuning is expensive; head-only capture most seed variance quickly.
Both report ROC-AUC, sensitivity, specificity, F1, MCC mean ± SD.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

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


# -------- Tabular --------

def _best_xgb_params():
    """Reuse best params from the earlier best-effort run."""
    p = ART / "pcos_best" / "results.json"
    r = json.load(open(p))
    return r["tuned_best_params"]["xgboost"]


def run_tabular_seed(seed, X, y, num_cols, cat_cols, params):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    clf = XGBClassifier(**params, eval_metric="logloss",
                        random_state=seed, tree_method="hist", n_jobs=-1)
    pipe = _pipe(make_preprocessor_tree(num_cols, cat_cols), clf)
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    y_prob = cal.predict_proba(X_te)[:, 1]
    m = full_metrics(y_te, y_prob)
    return {"seed": seed, "n_test": int(len(y_te)), **m}


def run_tabular():
    print(f"\n{'='*70}\nTabular multi-seed ({len(SEEDS)} seeds)\n{'='*70}")
    df = data_mod.load_all()
    feats = data_mod.get_feature_columns(df)
    num = [c for c in feats if c not in CATEGORICAL]
    cat = [c for c in feats if c in CATEGORICAL]
    X = df[feats]
    y = df["pcos"].astype(int).values
    params = _best_xgb_params()

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_tabular_seed(s, X, y, num, cat, params)
        rows.append(r)
        print(f"  seed={s}  AUC={r['roc_auc']:.4f}  Sens={r['sensitivity']:.3f}  "
              f"Spec={r['specificity']:.3f}  F1={r['f1']:.3f}  ({time.time()-t0:.1f}s)")

    return rows


# -------- CNN --------

IMG_SIZE = 224
BATCH = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_cnn():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    return m


def _loaders(seed):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ROOT = Path("analysis/data/ultrasound")
    train_full = datasets.ImageFolder(str(ROOT / "train"), transform=train_tf)
    n = len(train_full); n_val = int(0.15 * n)
    tr, va = random_split(train_full, [n - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    va.dataset = datasets.ImageFolder(str(ROOT / "train"), transform=eval_tf)
    va = torch.utils.data.Subset(va.dataset, va.indices)
    test = datasets.ImageFolder(str(ROOT / "test"), transform=eval_tf)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(va, batch_size=BATCH, shuffle=False),
            DataLoader(test, batch_size=BATCH, shuffle=False))


def _eval(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()  # infected = class 0 = PCOS+
            ys.append(y.numpy()); ps.append(p)
    y_true = (np.concatenate(ys) == 0).astype(int)  # flip so PCOS+ = 1
    return np.concatenate(ps), y_true


def run_cnn_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _loaders(seed)
    model = build_cnn().to(DEVICE)
    # Freeze backbone → head-only for speed
    for n, p in model.named_parameters():
        p.requires_grad = "classifier" in n
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    # Kaggle test
    p_int, y_int = _eval(model, te)
    m_int = full_metrics(y_int, p_int)

    # Zenodo external (truly-external subset)
    import pandas as pd
    zen_dir = Path("analysis/data/zenodo/images")
    lbl = pd.read_csv("analysis/data/zenodo/class label.csv")
    lbl.columns = [c.strip() for c in lbl.columns]
    lbl.rename(columns={
        "Class label (whether polycsytic ovary is visible or not visible)": "pco_label"
    }, inplace=True)
    lbl["pco_label"] = lbl["pco_label"].str.strip()
    lbl = lbl[lbl["imagePath"].apply(lambda p: (zen_dir / p).exists())]
    overlap = set(json.load(open(ART / "dedup" / "zenodo_overlap_names.json")))
    lbl = lbl[~lbl["imagePath"].isin(overlap)].reset_index(drop=True)

    from PIL import Image
    from torch.utils.data import Dataset
    class ZenDS(Dataset):
        def __init__(self, df, tf): self.df, self.tf = df, tf
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = Image.open(zen_dir / row["imagePath"]).convert("RGB")
            return self.tf(img), int(row["pco_label"] == "Visible")

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ze = DataLoader(ZenDS(lbl, eval_tf), batch_size=BATCH, shuffle=False)
    p_ext, y_ext = _eval(model, ze)
    m_ext = full_metrics(y_ext, p_ext)
    return {"seed": seed, "internal": m_int, "external_clean": m_ext,
            "n_internal": int(len(y_int)), "n_external": int(len(y_ext))}


def run_cnn():
    print(f"\n{'='*70}\nCNN multi-seed ({len(SEEDS)} seeds, head-only 3 epochs)\n{'='*70}")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_cnn_seed(s)
        rows.append(r)
        print(f"  seed={s}  int_AUC={r['internal']['roc_auc']:.4f}  "
              f"ext_AUC={r['external_clean']['roc_auc']:.4f}  "
              f"ext_Spec={r['external_clean']['specificity']:.3f}  "
              f"({time.time()-t0:.1f}s)")
    return rows


def summarize(rows, path_prefix):
    def agg(keys, get):
        out = {}
        for k in keys:
            arr = np.array([get(r, k) for r in rows])
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "values": [float(v) for v in arr.tolist()]}
        return out

    keys = ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]
    return {"seeds": [r["seed"] for r in rows], "aggregate": agg(keys, lambda r, k: r[k])}


def main():
    tab_rows = run_tabular()
    tab_summary = summarize(tab_rows, "tabular")
    print(f"\nTabular aggregate (mean ± SD across seeds):")
    for k, v in tab_summary["aggregate"].items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")

    cnn_rows = run_cnn()
    # For CNN, agg internal + external separately
    def agg_side(section):
        return {k: {"mean": float(np.mean([r[section][k] for r in cnn_rows])),
                    "std": float(np.std([r[section][k] for r in cnn_rows])),
                    "values": [float(r[section][k]) for r in cnn_rows]}
                for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]}
    cnn_summary = {"seeds": SEEDS,
                   "internal": agg_side("internal"),
                   "external_clean": agg_side("external_clean")}
    print(f"\nCNN internal aggregate (mean ± SD across seeds):")
    for k, v in cnn_summary["internal"].items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"\nCNN external (truly-external Zenodo) aggregate (mean ± SD across seeds):")
    for k, v in cnn_summary["external_clean"].items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")

    with open(OUT / "multiseed.json", "w") as f:
        json.dump({
            "tabular": {"rows": tab_rows, "summary": tab_summary},
            "cnn": {"rows": cnn_rows, "summary": cnn_summary},
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/multiseed.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
