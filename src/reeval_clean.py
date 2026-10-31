"""Re-evaluate CNN on truly-external Zenodo (removing Kaggle duplicates)
and on the ~9% of Kaggle-test images without train near-duplicates.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import bootstrap_ci, full_metrics

ART = Path("analysis/artifacts")
OUT = ART / "reeval_clean"
OUT.mkdir(exist_ok=True)


def main():
    # Load previous predictions
    ext_pco = pd.read_csv(ART / "external_val" / "preds_y_pco.csv")

    # Load overlap set
    overlap = set(json.load(open(ART / "dedup" / "zenodo_overlap_names.json")))
    print(f"Zenodo images that overlap with Kaggle: {len(overlap)}")

    # Split Zenodo predictions
    ext_pco["is_overlap"] = ext_pco["imagePath"].isin(overlap)
    clean = ext_pco[~ext_pco["is_overlap"]].copy()
    dirty = ext_pco[ext_pco["is_overlap"]].copy()

    print(f"\nZenodo predictions split:")
    print(f"  truly-external (no Kaggle near-dup): n={len(clean)} ({100*len(clean)/len(ext_pco):.1f}%)")
    print(f"  overlap-with-Kaggle              : n={len(dirty)} ({100*len(dirty)/len(ext_pco):.1f}%)")

    # Metrics on each subset
    print("\n=== Truly-external Zenodo subset ===")
    print(f"prevalence: {clean['y_true'].mean():.3f}")
    m_clean = full_metrics(clean["y_true"].values, clean["y_prob"].values)
    for k, v in m_clean.items():
        print(f"  {k:20s}  {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}  {v}")
    ci_clean = bootstrap_ci(clean["y_true"].values, clean["y_prob"].values,
                            n_iter=1000, seed=42)
    print(f"\nBootstrap 95% CIs (n={len(clean)}):")
    for k, v in ci_clean.items():
        print(f"  {k:20s}  {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    if len(dirty) > 5:
        print("\n=== Overlap subset (near-duplicates of Kaggle) ===")
        print(f"prevalence: {dirty['y_true'].mean():.3f}")
        m_dirty = full_metrics(dirty["y_true"].values, dirty["y_prob"].values)
        for k, v in m_dirty.items():
            print(f"  {k:20s}  {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}  {v}")

    with open(OUT / "reeval.json", "w") as f:
        json.dump({
            "n_clean": int(len(clean)),
            "n_dirty": int(len(dirty)),
            "clean_metrics": m_clean,
            "clean_ci": ci_clean,
            "dirty_metrics": full_metrics(dirty["y_true"].values, dirty["y_prob"].values) if len(dirty) > 5 else None,
        }, f, indent=2, default=float)
    print(f"\n→ {OUT}/reeval.json")


if __name__ == "__main__":
    main()
