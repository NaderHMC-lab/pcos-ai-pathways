"""Train a CNN classifier on the anaghachoudhari PCOS ultrasound dataset.

Architecture: MobileNetV3-Small pretrained on ImageNet, replaced classifier head.
Frozen backbone (fast on Apple MPS), train head for a few epochs, then fine-tune whole net.
Two-phase training keeps runtime under ~5 min on M-series Mac.

Also emits an internal (held-out ~20% of train) validation curve, then full test-set eval.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, accuracy_score, brier_score_loss,
)

warnings.filterwarnings("ignore")

DATA = Path("analysis/data/ultrasound")
ART = Path("analysis/artifacts/cnn_ultrasound")
ART.mkdir(exist_ok=True, parents=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
BATCH = 64
NUM_WORKERS = 0  # MPS + workers can conflict


def make_loaders():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    normalize = weights.transforms()  # includes resize + normalize

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(str(DATA / "train"), transform=train_tf)
    # ImageFolder class-to-idx alphabetical: infected=0, notinfected=1 → flip for "PCOS positive = 1"
    print(f"class_to_idx = {train_full.class_to_idx}")
    n = len(train_full)
    n_val = int(0.15 * n); n_tr = n - n_val
    tr, va = random_split(train_full, [n_tr, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    # eval transform for val
    va.dataset = datasets.ImageFolder(str(DATA / "train"), transform=eval_tf)
    va = torch.utils.data.Subset(va.dataset, va.indices)

    test = datasets.ImageFolder(str(DATA / "test"), transform=eval_tf)

    train_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    val_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    test_ld = DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    print(f"train={len(tr)}  val={len(va)}  test={len(test)}")
    return train_ld, val_ld, test_ld, train_full.class_to_idx


def build_model():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=weights)
    # Replace classifier for 2-class output
    in_feat = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feat, 2)
    return m


def freeze_backbone(model, freeze=True):
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = not freeze


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    probs_all, labels_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            probs = torch.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()  # infected=class 0
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            probs_all.append(probs); labels_all.append(y.detach().cpu().numpy())
    y_prob = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all)
    # flip labels so PCOS-positive = 1 (paper convention)
    y_true_pos = (y_true == 0).astype(int)  # infected → 1
    metrics = {
        "loss": loss_sum / total,
        "accuracy": accuracy_score(y_true_pos, (y_prob >= 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_true_pos, y_prob),
    }
    return metrics, y_prob, y_true_pos


def full_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train():
    print(f"Device: {DEVICE}")
    train_ld, val_ld, test_ld, class_map = make_loaders()

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ------ Phase 1: train head only ------
    print("\n[Phase 1] Head-only training (backbone frozen)")
    freeze_backbone(model, True)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=1e-3, weight_decay=1e-4)
    hist = []
    for epoch in range(3):
        t0 = time.time()
        tr_m, _, _ = run_epoch(model, train_ld, criterion, opt)
        va_m, _, _ = run_epoch(model, val_ld, criterion, None)
        hist.append({"phase": 1, "epoch": epoch, **{f"tr_{k}": v for k, v in tr_m.items()},
                     **{f"va_{k}": v for k, v in va_m.items()}, "time": time.time() - t0})
        print(f"  ep{epoch}  tr_loss={tr_m['loss']:.3f}  tr_auc={tr_m['roc_auc']:.3f}  "
              f"va_loss={va_m['loss']:.3f}  va_auc={va_m['roc_auc']:.3f}  ({hist[-1]['time']:.1f}s)")

    # ------ Phase 2: unfreeze last block(s) ------
    print("\n[Phase 2] Fine-tuning full model")
    freeze_backbone(model, False)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    best_va_auc = 0.0
    for epoch in range(5):
        t0 = time.time()
        tr_m, _, _ = run_epoch(model, train_ld, criterion, opt)
        va_m, _, _ = run_epoch(model, val_ld, criterion, None)
        scheduler.step()
        hist.append({"phase": 2, "epoch": epoch, **{f"tr_{k}": v for k, v in tr_m.items()},
                     **{f"va_{k}": v for k, v in va_m.items()}, "time": time.time() - t0})
        print(f"  ep{epoch}  tr_loss={tr_m['loss']:.3f}  tr_auc={tr_m['roc_auc']:.3f}  "
              f"va_loss={va_m['loss']:.3f}  va_auc={va_m['roc_auc']:.3f}  ({hist[-1]['time']:.1f}s)")
        if va_m["roc_auc"] > best_va_auc:
            best_va_auc = va_m["roc_auc"]
            torch.save(model.state_dict(), ART / "best_model.pt")
            print(f"     ★ new best (val_auc={va_m['roc_auc']:.3f}) — saved")

    # ------ Final test ------
    print("\n[Test] Reloading best-val checkpoint and evaluating on held-out test:")
    model.load_state_dict(torch.load(ART / "best_model.pt"))
    _, y_prob, y_true = run_epoch(model, test_ld, criterion, None)
    metrics = full_metrics(y_true, y_prob)
    for k, v in metrics.items():
        if k != "confusion":
            print(f"  {k:15s}  {v:.4f}")
    print(f"  confusion (rows y_true, cols y_pred): {metrics['confusion']}")

    # ------ Save ------
    np.save(ART / "test_probs.npy", y_prob)
    np.save(ART / "test_labels.npy", y_true)
    with open(ART / "results.json", "w") as f:
        json.dump({
            "device": str(DEVICE),
            "architecture": "mobilenet_v3_small",
            "img_size": IMG_SIZE,
            "class_to_idx": class_map,
            "class_convention": "y=1 means PCOS-positive (infected)",
            "history": hist,
            "best_val_auc": float(best_va_auc),
            "test_metrics": metrics,
        }, f, indent=2, default=float)
    print(f"\n→ {ART}/")


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
