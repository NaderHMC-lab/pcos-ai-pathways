"""Multi-seed CNN variance (head-only 3 epochs on frozen backbone)."""
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from metrics import full_metrics

ART = Path("analysis/artifacts")
DATA_KAG = Path("analysis/data/ultrasound")
DATA_ZEN = Path("analysis/data/zenodo/images")
ZEN_LBL = Path("analysis/data/zenodo/class label.csv")
OUT = ART / "multiseed"
OUT.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG = 224; BATCH = 64
SEEDS = [42, 43, 44, 45, 46]

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG + 32, IMG + 32)),
    transforms.RandomCrop(IMG), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ZenDS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df, tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DATA_ZEN / row["imagePath"]).convert("RGB")
        return self.tf(img), int(row["pco_label"] == "Visible")


def load_zen_clean():
    lbl = pd.read_csv(ZEN_LBL)
    lbl.columns = [c.strip() for c in lbl.columns]
    lbl.rename(columns={
        "Class label (whether polycsytic ovary is visible or not visible)": "pco_label",
    }, inplace=True)
    lbl["pco_label"] = lbl["pco_label"].str.strip()
    lbl = lbl[lbl["imagePath"].apply(lambda p: (DATA_ZEN / p).exists())]
    overlap = set(json.load(open(ART / "dedup" / "zenodo_overlap_names.json")))
    return lbl[~lbl["imagePath"].isin(overlap)].reset_index(drop=True)


def _eval(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            ys.append(y.numpy()); ps.append(p)
    y_true = (np.concatenate(ys) == 0).astype(int)
    return np.concatenate(ps), y_true


def run_seed(seed, zen_df):
    torch.manual_seed(seed); np.random.seed(seed)
    # Loaders
    train_full = datasets.ImageFolder(str(DATA_KAG / "train"), transform=TRAIN_TF)
    n = len(train_full); n_val = int(0.15 * n)
    tr, _ = random_split(train_full, [n - n_val, n_val],
                         generator=torch.Generator().manual_seed(seed))
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True)
    test = datasets.ImageFolder(str(DATA_KAG / "test"), transform=EVAL_TF)
    te_ld = DataLoader(test, batch_size=BATCH, shuffle=False)
    zen_ld = DataLoader(ZenDS(zen_df, EVAL_TF), batch_size=BATCH, shuffle=False)

    # Model — head-only training
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m = m.to(DEVICE)
    for n, p in m.named_parameters():
        p.requires_grad = "classifier" in n

    opt = optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(3):
        m.train()
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()

    # Evaluate
    p_int, y_int = _eval(m, te_ld)
    m_int = full_metrics(y_int, p_int)
    p_ext, y_ext = _eval(m, zen_ld)
    m_ext = full_metrics(y_ext, p_ext)
    return {"seed": seed,
            "internal": {k: float(v) for k, v in m_int.items()},
            "external_clean": {k: float(v) for k, v in m_ext.items()},
            "n_int": int(len(y_int)), "n_ext": int(len(y_ext))}


def main():
    zen_df = load_zen_clean()
    print(f"Zenodo (truly-external) n={len(zen_df)}  "
          f"prevalence={(zen_df['pco_label'] == 'Visible').mean():.3f}")

    print(f"\nCNN multi-seed ({len(SEEDS)} seeds, head-only 3 epochs)")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s, zen_df)
        rows.append(r)
        print(f"  seed={s}  int_AUC={r['internal']['roc_auc']:.4f}  "
              f"ext_AUC={r['external_clean']['roc_auc']:.4f}  "
              f"ext_Spec={r['external_clean']['specificity']:.3f}  "
              f"({time.time()-t0:.1f}s)")

    def agg(section):
        return {k: {"mean": float(np.mean([r[section][k] for r in rows])),
                    "std": float(np.std([r[section][k] for r in rows])),
                    "values": [r[section][k] for r in rows]}
                for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]}

    summary = {"seeds": SEEDS, "internal": agg("internal"),
               "external_clean": agg("external_clean")}
    print("\nInternal (Kaggle test — contaminated) aggregate:")
    for k, v in summary["internal"].items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")
    print("\nExternal (truly-external Zenodo) aggregate:")
    for k, v in summary["external_clean"].items():
        print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")

    with open(OUT / "multiseed_cnn.json", "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2, default=float)
    print(f"\n→ {OUT}/multiseed_cnn.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
