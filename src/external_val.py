"""External validation of the CNN on the Zenodo ultrasound cohort.

Loads the CNN trained on anaghachoudhari, evaluates on Zenodo images using the
"polycystic ovary visible / not-visible" label as the closest analog to the
Kaggle "infected / notinfected" split.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, accuracy_score, brier_score_loss,
)

warnings.filterwarnings("ignore")

ZEN_DIR = Path("analysis/data/zenodo")
ART_CNN = Path("analysis/artifacts/cnn_ultrasound")
OUT = Path("analysis/artifacts/external_val")
OUT.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224


class ZenodoDataset(Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["imagePath"]).convert("RGB")
        x = self.transform(img)
        y = int(row[self.label_col])
        return x, y


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
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }


def load_zenodo_labels():
    lbl_path = ZEN_DIR / "class label.csv"
    df = pd.read_csv(lbl_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Class label (whether the ultrasound images appears normal or has some abnormality)": "label_normal",
        "Class label (whether polycsytic ovary is visible or not visible)": "label_pco",
    }, inplace=True)
    df["label_normal"] = df["label_normal"].str.strip()
    df["label_pco"] = df["label_pco"].str.strip()
    # PCOS-positive = 1 (matches training convention: infected=1)
    df["y_pco"] = (df["label_pco"] == "Visible").astype(int)
    df["y_abnormal"] = (df["label_normal"] == "Appears abnormal").astype(int)
    # Keep only rows where the image actually exists
    exists = df["imagePath"].apply(lambda p: (ZEN_DIR / "images" / p).exists())
    df = df[exists].reset_index(drop=True)
    return df


def load_cnn():
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(torch.load(ART_CNN / "best_model.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


def infer(model, loader):
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            # class 0 = infected in training, that = PCOS-positive prob
            prob = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            ps.append(prob); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main():
    t0 = time.time()
    print("="*70, "\n  EXTERNAL VALIDATION on Zenodo 14591782\n", "="*70)

    df = load_zenodo_labels()
    print(f"Zenodo images with labels: {len(df)}")
    print(f"  PCO-visible = 1 (positive): {int(df['y_pco'].sum())} ({df['y_pco'].mean():.3f})")
    print(f"  Appears abnormal = 1     : {int(df['y_abnormal'].sum())} ({df['y_abnormal'].mean():.3f})")

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_cnn()

    results = {}
    for label_col in ["y_pco", "y_abnormal"]:
        ds = ZenodoDataset(df, ZEN_DIR / "images", eval_tf, label_col)
        ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        print(f"\n[{label_col}]")
        y_prob, y_true = infer(model, ld)
        m = full_metrics(y_true, y_prob)
        for k, v in m.items():
            if k != "confusion":
                print(f"  {k:15s}  {v:.4f}" if isinstance(v, float) else f"  {k:15s}  {v}")
        print(f"  confusion (rows y_true, cols y_pred): {m['confusion']}")
        results[label_col] = m
        # save per-image predictions
        pd.DataFrame({"imagePath": df["imagePath"], "y_true": y_true, "y_prob": y_prob}).to_csv(
            OUT / f"preds_{label_col}.csv", index=False)

    # Internal vs external comparison table
    with open(ART_CNN / "results.json") as f:
        internal = json.load(f)["test_metrics"]

    gap_pco = internal["roc_auc"] - results["y_pco"]["roc_auc"]
    gap_ab = internal["roc_auc"] - results["y_abnormal"]["roc_auc"]

    print(f"\n{'-'*70}")
    print(f"Internal test ROC-AUC  (anaghachoudhari test split): {internal['roc_auc']:.4f}")
    print(f"External ROC-AUC       (Zenodo, PCO-visible label):  {results['y_pco']['roc_auc']:.4f}   Δ = {gap_pco:+.4f}")
    print(f"External ROC-AUC       (Zenodo, abnormal label):     {results['y_abnormal']['roc_auc']:.4f}   Δ = {gap_ab:+.4f}")

    with open(OUT / "external_val.json", "w") as f:
        json.dump({
            "n_zenodo": int(len(df)),
            "internal_test_metrics": internal,
            "external_metrics": results,
            "roc_auc_gap": {"pco": float(gap_pco), "abnormal": float(gap_ab)},
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/  ({(time.time()-t0):.1f}s)")


if __name__ == "__main__":
    main()
