"""Load, clean, and engineer features for the Kottarathil PCOS dataset."""
import numpy as np
import pandas as pd

from config import RAW_XLSX, RENAME, PLAUSIBILITY, BINARY_YN, CATEGORICAL


def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def load_raw():
    df = pd.read_excel(RAW_XLSX, sheet_name="Full_new")
    if "Unnamed: 44" in df.columns:
        df = df.drop(columns=["Unnamed: 44"])
    df.columns = [c.rstrip() for c in df.columns]
    df = df.rename(columns=RENAME)
    missing = set(RENAME.values()) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing after rename: {missing}")
    return df


def clean(df):
    df = df.copy()
    for c in ["ii_beta_hcg_miu_ml", "amh_ng_ml", "fast_food_y_n"]:
        if c in df.columns:
            df[c] = _to_numeric(df[c])
    for c, (lo, hi) in PLAUSIBILITY.items():
        if c in df.columns:
            v = _to_numeric(df[c])
            df[c] = v.where(v.between(lo, hi))
    for c in BINARY_YN:
        if c in df.columns:
            v = _to_numeric(df[c])
            df[c] = v.where(v.isin([0, 1]))
    for c in ["sl_no", "patient_no"]:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def _bmi_category(bmi):
    bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(bmi, bins=bins, labels=labels).astype("float")


def engineer(df):
    df = df.copy()
    df["bmi_category"] = _bmi_category(df["bmi"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lh_fsh_ratio"] = df["lh_miu_ml"] / df["fsh_miu_ml"]
    if "waist_hip_ratio" not in df.columns:
        df["waist_hip_ratio"] = df["waist_in"] / df["hip_in"]
    df["homa_ir"] = np.nan
    df["mr_central_obesity"] = (df["waist_in"] >= 31.5).astype(float)
    df.loc[df["waist_in"].isna(), "mr_central_obesity"] = np.nan
    bp_flag = ((df["bp_sys"] >= 130) | (df["bp_dia"] >= 85)).astype(float)
    bp_flag[df["bp_sys"].isna() & df["bp_dia"].isna()] = np.nan
    df["mr_high_bp"] = bp_flag
    df["mr_weight_gain"] = df["weight_gain_y_n"].astype(float)
    df["mr_diet_risk"] = df["fast_food_y_n"].astype(float)
    df["metabolic_risk_score"] = (
        df[["mr_central_obesity", "mr_high_bp", "mr_weight_gain", "mr_diet_risk"]]
        .sum(axis=1, skipna=True)
    )
    return df


def derive_pmos(df):
    d = df.copy()
    d["pmos"] = ((d["pcos"] == 1) & (d["metabolic_risk_score"] >= 3)).astype(int)
    return d


def get_feature_columns(df):
    exclude = {"pcos", "pmos", "homa_ir"}
    return [c for c in df.columns if c not in exclude]


def load_all():
    df = load_raw()
    df = clean(df)
    df = engineer(df)
    df = derive_pmos(df)
    return df
