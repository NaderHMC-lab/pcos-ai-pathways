"""Multi-label CNN trained on Zenodo directly.

Two heads, one shared backbone:
  1. PCO-visible / not-visible
  2. Appears-abnormal / normal

Zenodo split 70/15/15 (train/val/test), full 2-phase fine-tuning.
Reports per-head metrics and joint metrics.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score, accuracy_score, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from metrics import bootstrap_ci

ART = Path("analysis/artifacts")
OUT = ART / "multilabel_cnn"
OUT.mkdir(exist_ok=True, parents=True)
DATA = Path("analysis/data/zenodo")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG = 224
BATCH = 64
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG + 32, IMG + 32)),
    transforms.RandomCrop(IMG), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ZenMLDataset(Dataset):
    """Returns (image, {y_pco, y_abn}) tuples."""
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DATA / "images" / row["imagePath"]).convert("RGB")
        return (self.tf(img),
                torch.tensor(int(row["y_pco"]), dtype=torch.long),
                torch.tensor(int(row["y_abn"]), dtype=torch.long))


class MultiHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Keep backbone, drop original classifier
        self.backbone = base.features
        self.avgpool = base.avgpool
        # Replicate dropout + hidden linear from original MobileNet
        in_c = 576  # last block output channels for MobileNetV3-Small
        hidden = 1024
        self.shared = nn.Sequential(
            nn.Linear(in_c, hidden), nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
        )
        self.head_pco = nn.Linear(hidden, 2)
        self.head_abn = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.backbone(x); x = self.avgpool(x); x = torch.flatten(x, 1)
        s = self.shared(x)
        return self.head_pco(s), self.head_abn(s)


def load_zen():
    df = pd.read_csv(DATA / "class label.csv")
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Class label (whether the ultrasound images appears normal or has some abnormality)": "abn",
        "Class label (whether polycsytic ovary is visible or not visible)": "pco",
    }, inplace=True)
    df["abn"] = df["abn"].str.strip(); df["pco"] = df["pco"].str.strip()
    df["y_pco"] = (df["pco"] == "Visible").astype(int)
    df["y_abn"] = (df["abn"] == "Appears abnormal").astype(int)
    df = df[df["imagePath"].apply(lambda p: (DATA / "images" / p).exists())].reset_index(drop=True)
    return df


def train_epoch(model, loader, opt, crit):
    model.train(); loss_sum = 0.0; n = 0
    for x, y1, y2 in loader:
        x, y1, y2 = x.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE)
        l1, l2 = model(x)
        loss = crit(l1, y1) + crit(l2, y2)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * y1.size(0); n += y1.size(0)
    return loss_sum / n


def eval_epoch(model, loader):
    model.eval()
    ps1, ps2, ys1, ys2 = [], [], [], []
    with torch.no_grad():
        for x, y1, y2 in loader:
            x = x.to(DEVICE)
            l1, l2 = model(x)
            ps1.append(torch.softmax(l1, dim=1)[:, 1].cpu().numpy())
            ps2.append(torch.softmax(l2, dim=1)[:, 1].cpu().numpy())
            ys1.append(y1.numpy()); ys2.append(y2.numpy())
    return (np.concatenate(ys1), np.concatenate(ps1),
            np.concatenate(ys2), np.concatenate(ps2))


def head_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    df = load_zen()
    print(f"Zenodo n={len(df)}  "
          f"PCO-visible prevalence={df['y_pco'].mean():.3f}  "
          f"abnormal prevalence={df['y_abn'].mean():.3f}")

    # 70/15/15 split with joint stratification via y_pco (primary label)
    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["y_pco"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["y_pco"], random_state=SEED)
    print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")

    tr_ld = DataLoader(ZenMLDataset(tr, TRAIN_TF), batch_size=BATCH, shuffle=True)
    va_ld = DataLoader(ZenMLDataset(va, EVAL_TF), batch_size=BATCH, shuffle=False)
    te_ld = DataLoader(ZenMLDataset(te, EVAL_TF), batch_size=BATCH, shuffle=False)

    model = MultiHeadNet().to(DEVICE)
    crit = nn.CrossEntropyLoss()

    # Phase 1: head-only
    for n, p in model.named_parameters():
        p.requires_grad = ("shared" in n) or ("head_" in n)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    print("\n[Phase 1] Head-only (3 epochs)")
    for ep in range(3):
        t0 = time.time()
        tr_loss = train_epoch(model, tr_ld, opt, crit)
        y1, p1, y2, p2 = eval_epoch(model, va_ld)
        va_auc_pco = roc_auc_score(y1, p1); va_auc_abn = roc_auc_score(y2, p2)
        print(f"  ep{ep}  tr_loss={tr_loss:.3f}  "
              f"va_AUC_pco={va_auc_pco:.3f}  va_AUC_abn={va_auc_abn:.3f}  ({time.time()-t0:.1f}s)")

    # Phase 2: fine-tune whole network
    for p in model.parameters():
        p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    print("\n[Phase 2] Fine-tuning (5 epochs)")
    best_val = 0.0
    for ep in range(5):
        t0 = time.time()
        tr_loss = train_epoch(model, tr_ld, opt, crit)
        y1, p1, y2, p2 = eval_epoch(model, va_ld)
        va_auc_pco = roc_auc_score(y1, p1); va_auc_abn = roc_auc_score(y2, p2)
        sched.step()
        mean_va = (va_auc_pco + va_auc_abn) / 2
        star = ""
        if mean_va > best_val:
            best_val = mean_va
            torch.save(model.state_dict(), OUT / "best.pt")
            star = "★"
        print(f"  ep{ep}  tr_loss={tr_loss:.3f}  "
              f"va_AUC_pco={va_auc_pco:.3f}  va_AUC_abn={va_auc_abn:.3f}  "
              f"({time.time()-t0:.1f}s) {star}")

    # Test-set eval with best checkpoint
    model.load_state_dict(torch.load(OUT / "best.pt"))
    y1, p1, y2, p2 = eval_epoch(model, te_ld)
    m_pco = head_metrics(y1, p1); m_abn = head_metrics(y2, p2)
    ci_pco = bootstrap_ci(y1, p1, n_iter=1000, seed=SEED)
    ci_abn = bootstrap_ci(y2, p2, n_iter=1000, seed=SEED)

    print("\n" + "=" * 70)
    print("Zenodo held-out test results")
    print("=" * 70)
    print(f"\nHead 1 (PCO-visible)  n={len(y1)}  prevalence={y1.mean():.3f}")
    for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1"]:
        v = ci_pco[k]
        print(f"  {k:15s}  {m_pco[k]:.4f}   CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print(f"\nHead 2 (Appears abnormal)  n={len(y2)}  prevalence={y2.mean():.3f}")
    for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1"]:
        v = ci_abn[k]
        print(f"  {k:15s}  {m_abn[k]:.4f}   CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    # Save
    np.save(OUT / "test_labels_pco.npy", y1); np.save(OUT / "test_probs_pco.npy", p1)
    np.save(OUT / "test_labels_abn.npy", y2); np.save(OUT / "test_probs_abn.npy", p2)
    with open(OUT / "results.json", "w") as f:
        json.dump({
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "test_metrics": {"pco": m_pco, "abnormal": m_abn},
            "test_ci": {"pco": ci_pco, "abnormal": ci_abn},
            "best_val_mean_auc": float(best_val),
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
