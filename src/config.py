"""Column mappings, plausibility windows, and constants."""

RAW_XLSX = "analysis/data/PCOS_data_without_infertility.xlsx"

RENAME = {
    "Sl. No": "sl_no",
    "Patient File No.": "patient_no",
    "PCOS (Y/N)": "pcos",
    " Age (yrs)": "age_yrs",
    "Weight (Kg)": "weight_kg",
    "Height(Cm)": "height_cm",
    "BMI": "bmi",
    "Blood Group": "blood_group",
    "Pulse rate(bpm)": "pulse_bpm",
    "RR (breaths/min)": "rr",
    "Hb(g/dl)": "hb_g_dl",
    "Cycle(R/I)": "cycle_r_i",
    "Cycle length(days)": "cycle_length_d",
    "Marraige Status (Yrs)": "marriage_status_yrs",
    "Pregnant(Y/N)": "pregnant_y_n",
    "No. of aborptions": "n_abortions",
    "  I   beta-HCG(mIU/mL)": "i_beta_hcg_miu_ml",
    "II    beta-HCG(mIU/mL)": "ii_beta_hcg_miu_ml",
    "FSH(mIU/mL)": "fsh_miu_ml",
    "LH(mIU/mL)": "lh_miu_ml",
    "FSH/LH": "fsh_lh",
    "Hip(inch)": "hip_in",
    "Waist(inch)": "waist_in",
    "Waist:Hip Ratio": "waist_hip_ratio",
    "TSH (mIU/L)": "tsh_miu_l",
    "AMH(ng/mL)": "amh_ng_ml",
    "PRL(ng/mL)": "prl_ng_ml",
    "Vit D3 (ng/mL)": "vit_d3_ng_ml",
    "PRG(ng/mL)": "prg_ng_ml",
    "RBS(mg/dl)": "rbs_mg_dl",
    "Weight gain(Y/N)": "weight_gain_y_n",
    "hair growth(Y/N)": "hair_growth_y_n",
    "Skin darkening (Y/N)": "skin_darkening_y_n",
    "Hair loss(Y/N)": "hair_loss_y_n",
    "Pimples(Y/N)": "pimples_y_n",
    "Fast food (Y/N)": "fast_food_y_n",
    "Reg.Exercise(Y/N)": "reg_exercise_y_n",
    "BP _Systolic (mmHg)": "bp_sys",
    "BP _Diastolic (mmHg)": "bp_dia",
    "Follicle No. (L)": "follicle_no_l",
    "Follicle No. (R)": "follicle_no_r",
    "Avg. F size (L) (mm)": "avg_f_size_l_mm",
    "Avg. F size (R) (mm)": "avg_f_size_r_mm",
    "Endometrium (mm)": "endometrium_mm",
}

PLAUSIBILITY = {
    "age_yrs": (10, 60), "weight_kg": (25, 200), "height_cm": (120, 210),
    "bmi": (10, 60), "pulse_bpm": (30, 200), "rr": (8, 40), "hb_g_dl": (5, 20),
    "cycle_length_d": (1, 90), "marriage_status_yrs": (0, 40), "n_abortions": (0, 10),
    "i_beta_hcg_miu_ml": (0, 200000), "ii_beta_hcg_miu_ml": (0, 200000),
    "fsh_miu_ml": (0, 200), "lh_miu_ml": (0, 200), "fsh_lh": (0, 100),
    "hip_in": (20, 70), "waist_in": (18, 70), "waist_hip_ratio": (0.5, 1.5),
    "tsh_miu_l": (0, 200), "amh_ng_ml": (0, 100), "prl_ng_ml": (0, 500),
    "vit_d3_ng_ml": (0, 150), "prg_ng_ml": (0, 300), "rbs_mg_dl": (40, 500),
    "bp_sys": (70, 220), "bp_dia": (40, 140),
    "follicle_no_l": (0, 40), "follicle_no_r": (0, 40),
    "avg_f_size_l_mm": (0, 40), "avg_f_size_r_mm": (0, 40),
    "endometrium_mm": (0, 30),
}

BINARY_YN = ["pregnant_y_n", "weight_gain_y_n", "hair_growth_y_n", "skin_darkening_y_n",
             "hair_loss_y_n", "pimples_y_n", "fast_food_y_n", "reg_exercise_y_n"]
CATEGORICAL = ["blood_group", "cycle_r_i"]

RANDOM_STATE = 42
BOOT_ITERS = 1000
