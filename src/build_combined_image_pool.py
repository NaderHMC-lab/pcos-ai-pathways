"""Build a deduplicated combined image pool from 4 cohorts.

Sources:
  - anaghachoudhari train/test (Kaggle)  → PCOS-positive folder = 'infected'
  - PCOSGen-train (Zenodo 10430727)      → 'Healthy' == 0 means PCOS-positive
  - Zenodo 14591782 (external test set)  → 'PCO visible/not visible'

Output:
  - artifacts/pool/all_images.csv         with columns: path, source, y_pcos, phash_hex, phash_int
  - artifacts/pool/pool_dedup_stats.json  cohort sizes, cross-cohort duplicates
  - artifacts/pool/train.csv              training pool (deduped against external test)
  - artifacts/pool/external_test.csv      truly-external Zenodo 14591782 subset
"""
import json
from collections import defaultdict
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

DATA = Path("analysis/data")
OUT = Path("analysis/artifacts/pool")
OUT.mkdir(exist_ok=True, parents=True)

THRESHOLD = 5  # Hamming distance for near-duplicates


def phash(path):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"), hash_size=8)
    except Exception:
        return None


def scan_kaggle(root, source):
    """Kaggle: 'infected' = PCOS-positive (label 1), 'notinfected' = 0."""
    rows = []
    for cls in ["infected", "notinfected"]:
        for p in sorted((root / cls).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            h = phash(p)
            if h is None:
                continue
            rows.append({
                "path": str(p),
                "source": source,
                "y_pcos": 1 if cls == "infected" else 0,
                "phash_hex": str(h),
                "phash_int": int(str(h), 16),
            })
    return rows


def scan_pcosgen():
    """PCOSGen: label 'Healthy' 0 means PCOS-positive."""
    root = DATA / "pcosgen_train" / "PCOSGen-train"
    lbl = pd.read_excel(root / "class_label.xlsx")
    lbl.columns = [c.strip() for c in lbl.columns]
    rows = []
    for _, r in lbl.iterrows():
        p = root / "images" / r["imagePath"]
        if not p.exists():
            continue
        h = phash(p)
        if h is None:
            continue
        rows.append({
            "path": str(p),
            "source": "pcosgen_train",
            "y_pcos": int(r["Healthy"] == 0),
            "phash_hex": str(h),
            "phash_int": int(str(h), 16),
        })
    return rows


def scan_zenodo():
    root = DATA / "zenodo"
    lbl = pd.read_csv(root / "class label.csv")
    lbl.columns = [c.strip() for c in lbl.columns]
    lbl.rename(columns={
        "Class label (whether polycsytic ovary is visible or not visible)": "pco_label",
    }, inplace=True)
    lbl["pco_label"] = lbl["pco_label"].str.strip()
    rows = []
    for _, r in lbl.iterrows():
        p = root / "images" / r["imagePath"]
        if not p.exists():
            continue
        h = phash(p)
        if h is None:
            continue
        rows.append({
            "path": str(p),
            "source": "zenodo_14591782",
            "y_pcos": int(r["pco_label"] == "Visible"),
            "phash_hex": str(h),
            "phash_int": int(str(h), 16),
        })
    return rows


def hamming(a, b):
    return bin(a ^ b).count("1")


def dedup_within(rows, threshold=THRESHOLD):
    """Remove exact-hash duplicates within a set (keep first occurrence)."""
    seen = set()
    keep = []
    for r in rows:
        if r["phash_hex"] in seen:
            continue
        seen.add(r["phash_hex"])
        keep.append(r)
    return keep


def find_near_dupes(pool_a, pool_b, threshold=THRESHOLD):
    """Return list of (a_idx, b_idx, hamming) for cross-pool near-duplicates."""
    hits = []
    for i, ra in enumerate(pool_a):
        for j, rb in enumerate(pool_b):
            d = hamming(ra["phash_int"], rb["phash_int"])
            if d <= threshold:
                hits.append((i, j, d))
    return hits


def main():
    print("Scanning anaghachoudhari train/test…")
    kag_train = scan_kaggle(DATA / "ultrasound/train", "kaggle_train")
    kag_test = scan_kaggle(DATA / "ultrasound/test", "kaggle_test")
    print(f"  train={len(kag_train)}  test={len(kag_test)}")

    print("Scanning PCOSGen-train…")
    pcosgen = scan_pcosgen()
    print(f"  n={len(pcosgen)}")

    print("Scanning Zenodo 14591782…")
    zen = scan_zenodo()
    print(f"  n={len(zen)}")

    all_rows = kag_train + kag_test + pcosgen + zen
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT / "all_images.csv", index=False)
    print(f"\nTotal images: {len(df_all)}")

    # ---- Deduplicate exact within-cohort ----
    kag_all = dedup_within(kag_train + kag_test)
    pg = dedup_within(pcosgen)
    zn = dedup_within(zen)
    print(f"\nAfter within-cohort exact-hash dedup:")
    print(f"  anaghachoudhari:  {len(kag_all)}")
    print(f"  pcosgen_train:    {len(pg)}")
    print(f"  zenodo:           {len(zn)}")

    # ---- Cross-cohort near-duplicates ----
    print(f"\nSearching cross-cohort near-duplicates (Hamming ≤ {THRESHOLD})…")
    kp_hits = find_near_dupes(kag_all, pg)
    kz_hits = find_near_dupes(kag_all, zn)
    pz_hits = find_near_dupes(pg, zn)

    kp_uniq_k = len({i for i, _, _ in kp_hits})
    kp_uniq_p = len({j for _, j, _ in kp_hits})
    kz_uniq_k = len({i for i, _, _ in kz_hits})
    kz_uniq_z = len({j for _, j, _ in kz_hits})
    pz_uniq_p = len({i for i, _, _ in pz_hits})
    pz_uniq_z = len({j for _, j, _ in pz_hits})

    print(f"  kaggle ↔ pcosgen : {len(kp_hits)} pairs   "
          f"(unique kaggle={kp_uniq_k}, unique pcosgen={kp_uniq_p})")
    print(f"  kaggle ↔ zenodo  : {len(kz_hits)} pairs   "
          f"(unique kaggle={kz_uniq_k}, unique zenodo={kz_uniq_z})")
    print(f"  pcosgen ↔ zenodo : {len(pz_hits)} pairs   "
          f"(unique pcosgen={pz_uniq_p}, unique zenodo={pz_uniq_z})")

    # ---- Build training pool: kaggle ∪ pcosgen, minus anything overlapping Zenodo ----
    zen_dupe_kag = {i for i, _, _ in kz_hits}
    zen_dupe_pg = {i for i, _, _ in pz_hits}

    kag_clean = [r for i, r in enumerate(kag_all) if i not in zen_dupe_kag]
    pg_clean = [r for i, r in enumerate(pg) if i not in zen_dupe_pg]
    # Also dedup training between pcosgen and kaggle (keep pcosgen, drop kaggle dupes)
    pg_hashes = {r["phash_hex"] for r in pg_clean}
    kag_final = [r for r in kag_clean if r["phash_hex"] not in pg_hashes]
    # Recheck near-dup (post-exact-dedup): drop any remaining kag matching pg near
    kag_final_dedup = []
    pg_ints = [r["phash_int"] for r in pg_clean]
    for r in kag_final:
        if any(hamming(r["phash_int"], pi) <= THRESHOLD for pi in pg_ints):
            continue
        kag_final_dedup.append(r)

    train_pool = pg_clean + kag_final_dedup
    print(f"\nCombined training pool (dedup vs Zenodo and vs each other):")
    print(f"  pcosgen_train (kept):  {len(pg_clean)}")
    print(f"  kaggle (kept):         {len(kag_final_dedup)}")
    print(f"  TOTAL training pool:   {len(train_pool)}")
    print(f"  Class balance: PCOS+ = {sum(r['y_pcos'] for r in train_pool)}  "
          f"({sum(r['y_pcos'] for r in train_pool)/len(train_pool):.3f})")

    # ---- Truly-external Zenodo test set ----
    # Zenodo cleaned = no near-dup with kaggle OR pcosgen
    zen_dup_from_kag = {j for _, j, _ in kz_hits}
    zen_dup_from_pg = {j for _, j, _ in pz_hits}
    external = [r for i, r in enumerate(zn) if i not in (zen_dup_from_kag | zen_dup_from_pg)]
    print(f"\nTruly-external Zenodo test set:")
    print(f"  kept: {len(external)} / {len(zn)}  ({100*len(external)/len(zn):.1f}%)")
    print(f"  Class balance: PCOS+ = {sum(r['y_pcos'] for r in external)}  "
          f"({sum(r['y_pcos'] for r in external)/len(external):.3f})")

    pd.DataFrame(train_pool).to_csv(OUT / "train.csv", index=False)
    pd.DataFrame(external).to_csv(OUT / "external_test.csv", index=False)

    with open(OUT / "pool_dedup_stats.json", "w") as f:
        json.dump({
            "cohort_raw_sizes": {
                "kaggle_train": len(kag_train), "kaggle_test": len(kag_test),
                "pcosgen_train": len(pcosgen), "zenodo": len(zen),
            },
            "cohort_dedup_sizes": {
                "kaggle_all": len(kag_all), "pcosgen": len(pg), "zenodo": len(zn),
            },
            "cross_cohort": {
                "kaggle_pcosgen_pairs": len(kp_hits),
                "kaggle_pcosgen_unique_kag": kp_uniq_k, "kaggle_pcosgen_unique_pg": kp_uniq_p,
                "kaggle_zenodo_pairs": len(kz_hits),
                "kaggle_zenodo_unique_kag": kz_uniq_k, "kaggle_zenodo_unique_zen": kz_uniq_z,
                "pcosgen_zenodo_pairs": len(pz_hits),
                "pcosgen_zenodo_unique_pg": pz_uniq_p, "pcosgen_zenodo_unique_zen": pz_uniq_z,
            },
            "training_pool": {
                "pcosgen_kept": len(pg_clean),
                "kaggle_kept": len(kag_final_dedup),
                "total": len(train_pool),
                "prevalence_pcos_pos": float(sum(r["y_pcos"] for r in train_pool) / len(train_pool)),
            },
            "external_test": {
                "n": len(external),
                "prevalence_pcos_pos": float(sum(r["y_pcos"] for r in external) / len(external)),
            },
        }, f, indent=2)
    print(f"\n→ {OUT}/")


if __name__ == "__main__":
    main()
