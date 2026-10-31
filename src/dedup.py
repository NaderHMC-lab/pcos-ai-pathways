"""Cross-cohort perceptual-hash deduplication.

Computes pHash for every image in Kaggle train, Kaggle test, and Zenodo, then
finds cross-cohort near-duplicates (Hamming distance ≤ THRESHOLD).

Threshold rationale: with 64-bit pHash, distance 0 = identical byte-hash,
1-5 = perceptually identical or near-identical (crop, resize, mild recompression).
Distances above ~10 are safely dissimilar.
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

warnings.filterwarnings("ignore")

DATA = Path("analysis/data")
KAG = DATA / "ultrasound"
ZEN_DIR = DATA / "zenodo" / "images"
OUT = Path("analysis/artifacts/dedup")
OUT.mkdir(exist_ok=True, parents=True)

THRESHOLD = 5  # Hamming distance for near-duplicate


def scan_dir(dir_path, label_source):
    """Return list of {path, phash, dhash, source, class}."""
    out = []
    for p in sorted(dir_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        try:
            ph = imagehash.phash(img, hash_size=8)  # 64-bit pHash
            dh = imagehash.dhash(img, hash_size=8)
        except Exception:
            continue
        out.append({
            "path": str(p),
            "source": label_source,
            "class": p.parent.name if p.parent.parent == dir_path else "",
            "phash": str(ph),
            "phash_int": int(str(ph), 16),
            "dhash": str(dh),
        })
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    print("Scanning Kaggle train...")
    kag_train = scan_dir(KAG / "train", "kaggle_train")
    print(f"  {len(kag_train)} images")
    print("Scanning Kaggle test...")
    kag_test = scan_dir(KAG / "test", "kaggle_test")
    print(f"  {len(kag_test)} images")
    print("Scanning Zenodo...")
    zen = scan_dir(ZEN_DIR, "zenodo")
    print(f"  {len(zen)} images")

    all_imgs = kag_train + kag_test + zen

    # ---- 1. Exact pHash duplicates (Hamming = 0) ----
    exact = defaultdict(list)
    for im in all_imgs:
        exact[im["phash"]].append(im)
    exact_dupes = {k: v for k, v in exact.items() if len(v) > 1}
    print(f"\nExact pHash duplicates: {len(exact_dupes)} distinct hashes involving "
          f"{sum(len(v) for v in exact_dupes.values())} images.")

    # ---- 2. Near-duplicates across cohorts (Hamming ≤ THRESHOLD) ----
    # Efficient search: only cross-cohort pairs
    print(f"\nSearching cross-cohort near-duplicates (Hamming ≤ {THRESHOLD})...")
    zen_hashes = [(i["phash_int"], i) for i in zen]
    kag_hashes = [(i["phash_int"], i) for i in (kag_train + kag_test)]

    near_dup_zen_kag = []
    for zh, zi in zen_hashes:
        for kh, ki in kag_hashes:
            d = hamming(zh, kh)
            if d <= THRESHOLD:
                near_dup_zen_kag.append({
                    "hamming": d,
                    "zenodo_path": zi["path"],
                    "kaggle_path": ki["path"],
                    "kaggle_source": ki["source"],
                    "kaggle_class": ki["class"],
                })

    # Also within Kaggle test / train
    print("Searching Kaggle train ↔ test overlaps...")
    kt_hashes = [(i["phash_int"], i) for i in kag_train]
    kv_hashes = [(i["phash_int"], i) for i in kag_test]
    near_dup_train_test = []
    for kh, ki in kt_hashes:
        for vh, vi in kv_hashes:
            d = hamming(kh, vh)
            if d <= THRESHOLD:
                near_dup_train_test.append({
                    "hamming": d,
                    "train_path": ki["path"],
                    "test_path": vi["path"],
                    "train_class": ki["class"],
                    "test_class": vi["class"],
                })

    print(f"\n=== Cross-cohort near-duplicates ===")
    print(f"Zenodo ↔ Kaggle (any split):     {len(near_dup_zen_kag)}")
    print(f"Kaggle train ↔ Kaggle test :     {len(near_dup_train_test)}")

    # Class-conditional summary for train↔test
    if near_dup_train_test:
        cls_bd = defaultdict(int)
        for d in near_dup_train_test:
            cls_bd[f"{d['train_class']}→{d['test_class']}"] += 1
        print(f"\n  train→test class breakdown:")
        for k, v in sorted(cls_bd.items(), key=lambda x: -x[1]):
            print(f"    {k:30s}  {v}")

    # Zenodo ↔ Kaggle class breakdown
    if near_dup_zen_kag:
        cls_bd = defaultdict(int)
        for d in near_dup_zen_kag:
            cls_bd[d["kaggle_source"] + "/" + d["kaggle_class"]] += 1
        print(f"\n  Zenodo ↔ Kaggle breakdown:")
        for k, v in sorted(cls_bd.items(), key=lambda x: -x[1]):
            print(f"    {k:30s}  {v}")

    # Save
    with open(OUT / "dedup.json", "w") as f:
        json.dump({
            "counts": {
                "kaggle_train": len(kag_train),
                "kaggle_test": len(kag_test),
                "zenodo": len(zen),
            },
            "threshold": THRESHOLD,
            "exact_duplicates_count": sum(len(v) for v in exact_dupes.values()),
            "n_zenodo_kaggle_near_dupes": len(near_dup_zen_kag),
            "n_train_test_near_dupes": len(near_dup_train_test),
            "zenodo_kaggle_near_dupes": near_dup_zen_kag,
            "train_test_near_dupes": near_dup_train_test,
        }, f, indent=2)
    print(f"\n→ {OUT}/dedup.json")


if __name__ == "__main__":
    main()
