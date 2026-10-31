"""Grad-CAM diagnostic for the trained CNN.

Overlays class-activation heatmaps on ~12 images from each of:
  - anaghachoudhari test (internal): 6 infected + 6 notinfected
  - Zenodo (external):                6 PCO-visible + 6 not-visible

Goal: visually adjudicate whether the CNN attends to the ovary region
or to non-medical features (borders, watermarks, source artifacts).
"""
import warnings
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

warnings.filterwarnings("ignore")

DATA_KAG = Path("analysis/data/ultrasound/test")
DATA_ZEN = Path("analysis/data/zenodo/images")
ZEN_LBL = Path("analysis/data/zenodo/class label.csv")
ART_CNN = Path("analysis/artifacts/cnn_ultrasound")
FIGS = Path("analysis/figs")
FIGS.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG = 224
SEED = 42

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM on the last convolutional feature map."""

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, m, i, out):
        self.activations = out.detach()

    def _save_gradient(self, m, gin, gout):
        self.gradients = gout[0].detach()

    def close(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, x, target_class):
        """x: (1,3,H,W). Returns 224×224 heatmap in [0,1]."""
        self.model.zero_grad()
        logits = self.model(x)  # (1, num_classes)
        score = logits[:, target_class].sum()
        score.backward()

        # Weights: global-average-pool of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def build_model():
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(torch.load(ART_CNN / "best_model.pt", map_location=DEVICE))
    return m.to(DEVICE)


def _load_image_pair(path):
    img = Image.open(path).convert("RGB").resize((IMG, IMG))
    tens = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    return np.array(img) / 255.0, tens


def overlay(rgb, heatmap):
    """Blend a red-hot heatmap onto an RGB image."""
    cmap = plt.get_cmap("jet")
    hm_rgb = cmap(heatmap)[..., :3]
    return 0.55 * rgb + 0.45 * hm_rgb


def pick_kaggle(n_per_class=6):
    rng = Random(SEED)
    inf = sorted((DATA_KAG / "infected").glob("*"))
    nif = sorted((DATA_KAG / "notinfected").glob("*"))
    rng.shuffle(inf); rng.shuffle(nif)
    picks = []
    for p in inf[:n_per_class]:
        picks.append({"src": "kaggle_test", "true_label": "infected (PCOS+)", "path": p})
    for p in nif[:n_per_class]:
        picks.append({"src": "kaggle_test", "true_label": "notinfected (PCOS-)", "path": p})
    return picks


def pick_zenodo(n_per_class=6):
    df = pd.read_csv(ZEN_LBL)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Class label (whether polycsytic ovary is visible or not visible)": "pco_label",
    }, inplace=True)
    df["pco_label"] = df["pco_label"].str.strip()
    df = df[df["imagePath"].apply(lambda p: (DATA_ZEN / p).exists())]
    vis = df[df["pco_label"] == "Visible"].sample(n_per_class, random_state=SEED)
    inv = df[df["pco_label"] == "Not-visible"].sample(n_per_class, random_state=SEED)
    picks = []
    for _, r in vis.iterrows():
        picks.append({"src": "zenodo", "true_label": "PCO-visible (PCOS+)",
                      "path": DATA_ZEN / r["imagePath"]})
    for _, r in inv.iterrows():
        picks.append({"src": "zenodo", "true_label": "PCO-not-visible (PCOS-)",
                      "path": DATA_ZEN / r["imagePath"]})
    return picks


def run_grid(picks, model, target_layer, out_path, title):
    cam = GradCAM(model, target_layer)
    n = len(picks)
    ncols = 4
    nrows = int(np.ceil(n / ncols)) * 2  # each pick = 2 rows: original + overlay
    fig, axes = plt.subplots(nrows // 2, ncols * 2, figsize=(ncols * 4.5, (nrows // 2) * 2.2))
    axes = np.atleast_2d(axes)

    for idx, item in enumerate(picks):
        rgb, tens = _load_image_pair(item["path"])
        # Predict PCOS-positive prob (class 0 = infected)
        with torch.no_grad():
            logits = model(tens)
            prob_pos = torch.softmax(logits, dim=1)[0, 0].item()
        target_cls = 0 if prob_pos >= 0.5 else 1
        heat = cam(tens, target_cls)
        blend = overlay(rgb, heat)

        r = idx // ncols
        c_orig = (idx % ncols) * 2
        c_over = c_orig + 1

        axes[r, c_orig].imshow(rgb); axes[r, c_orig].axis("off")
        axes[r, c_orig].set_title(f"{item['true_label']}\np(PCOS+)={prob_pos:.2f}",
                                  fontsize=8)
        axes[r, c_over].imshow(blend); axes[r, c_over].axis("off")
        axes[r, c_over].set_title("Grad-CAM overlay", fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    cam.close()
    print(f"  → {out_path.name}")


def main():
    print("Loading trained CNN…")
    model = build_model()
    target_layer = model.features[-1]  # last conv-block output of MobileNetV3-Small

    print("Rendering Kaggle (internal) Grad-CAM grid…")
    run_grid(pick_kaggle(6), model, target_layer,
             FIGS / "gradcam_kaggle.png",
             "Grad-CAM — anaghachoudhari test images (internal)")

    print("Rendering Zenodo (external) Grad-CAM grid…")
    run_grid(pick_zenodo(6), model, target_layer,
             FIGS / "gradcam_zenodo.png",
             "Grad-CAM — Zenodo external images")


if __name__ == "__main__":
    main()
