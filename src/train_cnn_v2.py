"""Retrain CNN on the combined & deduplicated image pool.

Reads pool/train.csv (PCOSGen + deduped anaghachoudhari) and pool/external_test.csv
(truly-external Zenodo). 15% of the training pool is held out as an internal val split.

Architecture: MobileNetV3-Small pretrained on ImageNet, 2-phase transfer learning.

Reports internal val + external test metrics with bootstrap 95% CIs and
compares false-positive / false-negative counts to the original CNN.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from metrics import bootstrap_ci, full_metrics

ART = Path("analysis/artifacts")
POOL = ART / "pool"
OUT = ART / "cnn_v2"
OUT.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG = 224
BATCH = 64
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG + 32, IMG + 32)),
    transforms.RandomCrop(IMG),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PoolDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), int(row["y_pcos"])


def build_model():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    return m


def freeze_backbone(model, freeze=True):
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = not freeze


def train_epoch(model, loader, crit, opt):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()


def eval_split(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())  # class 1 = PCOS+
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    print("Loading pool CSVs…")
    train_all = pd.read_csv(POOL / "train.csv")
    ext = pd.read_csv(POOL / "external_test.csv")
    print(f"  training pool: n={len(train_all)}  PCOS+={train_all['y_pcos'].sum()}  "
          f"prev={train_all['y_pcos'].mean():.3f}")
    print(f"  external test: n={len(ext)}  PCOS+={ext['y_pcos'].sum()}  "
          f"prev={ext['y_pcos'].mean():.3f}")

    # 85/15 train/val stratified by (source, y_pcos)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_all, test_size=0.15,
        stratify=train_all[["source", "y_pcos"]].astype(str).agg("_".join, axis=1),
        random_state=SEED,
    )
    print(f"\n  train n={len(train_df)}   val n={len(val_df)}")

    tr_ld = DataLoader(PoolDataset(train_df, TRAIN_TF), batch_size=BATCH, shuffle=True)
    va_ld = DataLoader(PoolDataset(val_df, EVAL_TF), batch_size=BATCH, shuffle=False)
    te_ld = DataLoader(PoolDataset(ext, EVAL_TF), batch_size=BATCH, shuffle=False)

    model = build_model().to(DEVICE)
    crit = nn.CrossEntropyLoss()

    # Phase 1: head-only, 3 epochs
    print("\n[Phase 1] Head-only training (backbone frozen)")
    freeze_backbone(model, True)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=1e-3, weight_decay=1e-4)
    for ep in range(3):
        t0 = time.time()
        train_epoch(model, tr_ld, crit, opt)
        y_va, p_va = eval_split(model, va_ld)
        y_te, p_te = eval_split(model, te_ld)
        va_auc = roc_auc_score(y_va, p_va); te_auc = roc_auc_score(y_te, p_te)
        print(f"  ep{ep}  val_AUC={va_auc:.3f}  ext_AUC={te_auc:.3f}  ({time.time()-t0:.1f}s)")

    # Phase 2: full fine-tune, 6 epochs, cosine schedule
    print("\n[Phase 2] Fine-tuning full model")
    freeze_backbone(model, False)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    best_val_auc = 0.0
    for ep in range(6):
        t0 = time.time()
        train_epoch(model, tr_ld, crit, opt)
        y_va, p_va = eval_split(model, va_ld)
        y_te, p_te = eval_split(model, te_ld)
        sched.step()
        va_auc = roc_auc_score(y_va, p_va); te_auc = roc_auc_score(y_te, p_te)
        star = ""
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save(model.state_dict(), OUT / "best.pt")
            star = "★"
        print(f"  ep{ep}  val_AUC={va_auc:.3f}  ext_AUC={te_auc:.3f}  "
              f"({time.time()-t0:.1f}s) {star}")

    # Final eval with best checkpoint
    print(f"\n[Final] Reloading best-val checkpoint")
    model.load_state_dict(torch.load(OUT / "best.pt"))
    y_va, p_va = eval_split(model, va_ld)
    y_te, p_te = eval_split(model, te_ld)

    m_va = full_metrics(y_va, p_va); m_te = full_metrics(y_te, p_te)
    ci_va = bootstrap_ci(y_va, p_va, n_iter=1000, seed=SEED)
    ci_te = bootstrap_ci(y_te, p_te, n_iter=1000, seed=SEED)

    def show(name, m, ci):
        print(f"\n  {name}")
        for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
            print(f"    {k:15s}  {m[k]:.4f}   CI=[{ci[k]['ci_low']:.4f}, {ci[k]['ci_high']:.4f}]")

    show(f"Internal val (n={len(y_va)}):", m_va, ci_va)
    show(f"External test — truly-external Zenodo (n={len(y_te)}):", m_te, ci_te)

    # False positive / false negative counts
    def fp_fn(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                "fp_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "fn_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0}

    fp_fn_va = fp_fn(y_va, p_va)
    fp_fn_te = fp_fn(y_te, p_te)
    print(f"\n  Internal val FP/FN:  {fp_fn_va}")
    print(f"  External test FP/FN: {fp_fn_te}")

    np.save(OUT / "val_labels.npy", y_va); np.save(OUT / "val_probs.npy", p_va)
    np.save(OUT / "test_labels.npy", y_te); np.save(OUT / "test_probs.npy", p_te)
    with open(OUT / "results.json", "w") as f:
        json.dump({
            "n_train": len(train_df), "n_val": len(val_df), "n_external": len(ext),
            "prevalence_train": float(train_df["y_pcos"].mean()),
            "prevalence_val": float(val_df["y_pcos"].mean()),
            "prevalence_external": float(ext["y_pcos"].mean()),
            "val_metrics": m_va, "val_ci": ci_va, "val_fp_fn": fp_fn_va,
            "external_metrics": m_te, "external_ci": ci_te, "external_fp_fn": fp_fn_te,
            "best_val_auc": float(best_val_auc),
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
