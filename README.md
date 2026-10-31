# PCOS ultrasound — scripts and duplicate-aware evaluation manifests

This bundle contains the pipeline scripts and the deduplicated evaluation
manifests used in the PCOS ultrasound analysis. Everything is referenced
relative to the project root; run commands from the directory that
contains this `README.md`.

## Contents

```
src/                       # pipeline scripts (Python)
artifacts/
  dedup/
    dedup.json             # pHash duplicate map (threshold = 5)
    zenodo_overlap_names.json
  pool/
    all_images.csv         # every image + source + y_pcos + pHash
    train.csv              # deduped training pool (n = 3,172)
    external_test.csv      # Zenodo-only external test (n = 471)
    folds.csv              # 5-fold assignments, stratified by source × y_pcos
    pool_dedup_stats.json  # cohort/pair audit
  reeval_clean/
    reeval.json            # metrics on the clean splits
  cnn_v2/                  # primary CNN (ResNet-18 head), weights + val/test labels & probs
    best.pt
    results.json
    val_labels.npy  val_probs.npy  test_labels.npy  test_probs.npy
  cnn_v2_recal/
    recal.json             # temperature-scaled recalibration output for cnn_v2
  cnn_ultrasound/          # earlier ultrasound-only CNN (Kaggle train/test)
    best_model.pt
    results.json  test_labels.npy  test_probs.npy
  multilabel_cnn/          # dual-head CNN (PCOS + abnormal)
    best.pt
    results.json
    test_labels_pco.npy  test_probs_pco.npy
    test_labels_abn.npy  test_probs_abn.npy
  pcos_best/               # tabular model (XGBoost + isotonic calibration)
    calibrated_model.pkl  meta.pkl
    results.json  test_predictions.csv
  isotonic_recal/
    isotonic_recal.json    # isotonic recalibration output for pcos_best
  MODEL_WEIGHTS.sha256     # sha256 + size for every .pt / .pkl file
```

## Model weights

Five trained models are shipped in `artifacts/`:

| dir               | file                | arch / purpose                        | loader                 |
|-------------------|---------------------|---------------------------------------|------------------------|
| `cnn_v2/`         | `best.pt`           | ResNet-18 CNN, primary PCOS model     | `torch.load(..., map_location="cpu")`, see `src/train_cnn_v2.py` |
| `cnn_ultrasound/` | `best_model.pt`     | earlier Kaggle-only CNN               | `src/train_cnn.py`     |
| `multilabel_cnn/` | `best.pt`           | dual-head CNN (PCOS + abnormal)       | `src/multilabel_cnn.py`|
| `pcos_best/`      | `calibrated_model.pkl` + `meta.pkl` | tabular XGBoost + isotonic | `joblib.load(...)` / `pickle.load(...)`, see `src/run_pcos_best.py` |
| `cnn_v2_recal/` / `isotonic_recal/` | `*.json`  | recalibration parameters              | consumed by `src/recal_cnn_v2.py` / `src/isotonic_recal.py` |

Companion `*.npy` / `test_predictions.csv` files are the exact held-out
predictions reported in the manuscript — load them to reproduce every
metric in `artifacts/reeval_clean/reeval.json` without re-running training.

Verify integrity after download:

```bash
cd release/artifacts
# re-compute and compare
awk '{print $1"  "$3}' MODEL_WEIGHTS.sha256 | sed 's|^#.*||;/^$/d' > /tmp/want.txt
# then shasum -a 256 <each file> and diff
```

## What "duplicate-aware" means here

Three source cohorts (Kaggle train, Kaggle test, Zenodo, plus PCOSGen for
the pool build) overlap heavily — 4,153 exact duplicates and 2,903 near
duplicates (Hamming ≤ 5 on 64-bit perceptual hashes) were detected.
`artifacts/dedup/dedup.json` records every pair; `artifacts/pool/*.csv`
provide the resulting splits with:

- within-cohort duplicates collapsed to one representative
- cross-cohort pairs resolved so the external test (`external_test.csv`,
  Zenodo-only) is disjoint from the training pool (`train.csv`)
- 5-fold CV assignments (`folds.csv`) stratified on the composite
  `source × y_pcos` key — each fold holds ~54% PCOS+ and the same
  Kaggle/PCOSGen mix

All reported metrics (`artifacts/reeval_clean/reeval.json`) are computed on
these clean splits.

## Running the scripts

1. Place the raw datasets under `analysis/data/`:
   - `analysis/data/ultrasound/{train,test}/{infected,notinfected}/*.jpg`  (Kaggle)
   - `analysis/data/zenodo/images/*.jpg` + `analysis/data/zenodo/class label.csv`
   - `analysis/data/pcosgen_train/PCOSGen-train/images/*.jpg`
2. From the project root:
   ```bash
   python analysis/src/dedup.py                     # (re)build duplicate map
   python analysis/src/build_combined_image_pool.py # rebuild pool/*.csv
   python analysis/src/reeval_clean.py              # metrics on clean splits
   ```
3. CNN training / CV / recalibration:
   ```bash
   python analysis/src/train_cnn_v2.py
   python analysis/src/nested_cv.py
   python analysis/src/isotonic_recal.py
   python analysis/src/multiseed_cnn.py
   python analysis/src/external_val.py
   ```

## Manifest schemas

**`artifacts/pool/*.csv`**

| column    | type   | notes                                        |
|-----------|--------|----------------------------------------------|
| path      | str    | relative to project root                     |
| source    | str    | `kaggle_train`, `pcosgen_train`, `zenodo_14591782` |
| y_pcos    | 0/1    | PCOS label                                   |
| phash_hex | str    | 64-bit perceptual hash                       |
| phash_int | int    | same hash, decimal                           |
| fold      | 0..4   | *(folds.csv only)* 5-fold assignment         |

**`artifacts/dedup/dedup.json`** — top-level keys:

- `counts` — raw cohort sizes
- `threshold` — Hamming threshold (= 5)
- `exact_duplicates_count` — pairs with Hamming 0
- `n_zenodo_kaggle_near_dupes`, `n_train_test_near_dupes`
- `zenodo_kaggle_near_dupes[]` — `{hamming, zenodo_path, kaggle_path, kaggle_source, kaggle_class}`
- `train_test_near_dupes[]` — `{hamming, train_path, test_path, train_class, test_class}`

## Reproducibility notes

- pHash: 64-bit, Hamming threshold 5.
- CV folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the composite `source + "_" + y_pcos` key.
- Train/val split inside `train_cnn_v2.py`: 85/15, same composite stratifier.

## Paths

All paths in scripts and manifests are repo-relative. If you move the
directory, nothing needs to change — just invoke from the directory that
holds this README.
