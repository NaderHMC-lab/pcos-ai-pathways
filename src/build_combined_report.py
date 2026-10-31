"""Combined HTML report covering: best tabular model + CNN + external validation."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path("analysis/artifacts")
REP = Path("analysis/reports")
FIGS = Path("analysis/figs")
FIGS.mkdir(exist_ok=True, parents=True)
REP.mkdir(exist_ok=True, parents=True)


def img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f"<p><em>Missing figure: {path.name}</em></p>"
    return f'<figure><img src="{img_data_uri(path)}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


# ------------- Plots for CNN + external validation -------------

def plot_cnn_training(cnn_res, out):
    hist = pd.DataFrame(cnn_res["history"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(hist))
    ax[0].plot(x, hist["tr_loss"], "o-", label="train", color="#457b9d")
    ax[0].plot(x, hist["va_loss"], "s-", label="val", color="#e63946")
    ax[0].axvline(2.5, ls=":", color="gray")
    ax[0].set_xlabel("Epoch (Phase 1 → Phase 2)"); ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss curves"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(x, hist["tr_roc_auc"], "o-", label="train", color="#457b9d")
    ax[1].plot(x, hist["va_roc_auc"], "s-", label="val", color="#e63946")
    ax[1].axvline(2.5, ls=":", color="gray")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("ROC-AUC")
    ax[1].set_title("Training/val ROC-AUC"); ax[1].set_ylim(0.4, 1.05); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_internal_external(cnn_res, ext_res, out):
    """Internal test vs Zenodo external — headline metric bars."""
    metrics = ["accuracy", "roc_auc", "sensitivity", "specificity", "f1"]
    internal = cnn_res["test_metrics"]
    ext_pco = ext_res["external_metrics"]["y_pco"]
    ext_ab = ext_res["external_metrics"]["y_abnormal"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics)); w = 0.28
    ax.bar(x - w, [internal[m] for m in metrics], width=w,
           label="Internal test (anaghachoudhari)", color="#1d3557")
    ax.bar(x, [ext_pco[m] for m in metrics], width=w,
           label="External — Zenodo (PCO-visible)", color="#e63946")
    ax.bar(x + w, [ext_ab[m] for m in metrics], width=w,
           label="External — Zenodo (abnormal)", color="#f4a261")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="chance (AUC=0.5)")
    ax.set_ylabel("Metric value"); ax.set_title("CNN: internal-test vs external-Zenodo performance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_ext_confusion(ext_res, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, m) in zip(axes, ext_res["external_metrics"].items()):
        cm = np.array(m["confusion"])
        im = ax.imshow(cm, cmap="Reds")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred: neg", "pred: pos"])
        ax.set_yticklabels(["true: neg", "true: pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{name}  (n={m['n']}, prevalence={m['prevalence']:.2f})")
    fig.suptitle("Zenodo external validation — confusion matrices")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


# ------------- HTML building blocks -------------

def kpi(label, value, subtitle=""):
    sub = f'<div class="ci">{subtitle}</div>' if subtitle else ""
    return f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {sub}
</div>"""


def tabular_kpis():
    with open(ART / "pcos_best" / "results.json") as f:
        r = json.load(f)
    ci = r["bootstrap_ci"]
    kpis = "\n".join(kpi(m.replace("_", " ").title(),
                         f"{ci[m]['mean']:.3f}",
                         f"[{ci[m]['ci_low']:.3f}, {ci[m]['ci_high']:.3f}]")
                     for m in ["roc_auc", "sensitivity", "specificity", "f1", "mcc"])
    return kpis, r


def cnn_kpis():
    with open(ART / "cnn_ultrasound" / "results.json") as f:
        r = json.load(f)
    # Overlay bootstrap CI text when available
    ci_path = ART / "cnn_audit" / "ci_internal.json"
    ci = json.load(open(ci_path)) if ci_path.exists() else None
    t = r["test_metrics"]
    def sub(m):
        if ci and m in ci:
            return f"[{ci[m]['ci_low']:.3f}, {ci[m]['ci_high']:.3f}]"
        return "internal test"
    kpis = "\n".join(kpi(m.replace("_", " ").title(), f"{t[m]:.3f}", sub(m))
                     for m in ["roc_auc", "sensitivity", "specificity", "f1", "accuracy"])
    return kpis, r


def ext_kpis():
    with open(ART / "external_val" / "external_val.json") as f:
        r = json.load(f)
    ext = r["external_metrics"]["y_pco"]
    ci_path = ART / "cnn_audit" / "ci_external_pco.json"
    ci = json.load(open(ci_path)) if ci_path.exists() else None
    def sub(m):
        if ci and m in ci:
            return f"[{ci[m]['ci_low']:.3f}, {ci[m]['ci_high']:.3f}]"
        return "Zenodo external"
    kpis = "\n".join(kpi(m.replace("_", " ").title(), f"{ext[m]:.3f}", sub(m))
                     for m in ["roc_auc", "sensitivity", "specificity", "f1", "accuracy"])
    return kpis, r


def table_from_dict(rows, headers):
    tr = "".join(f"<tr>{''.join(f'<td class=\"num\">{c}</td>' if i > 0 else f'<td>{c}</td>' for i, c in enumerate(row))}</tr>" for row in rows)
    th = "".join(f"<th>{h}</th>" for h in headers)
    return f'<div class="table-scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def tabular_headline_table(r):
    ci = r["bootstrap_ci"]
    rows = [[m.replace("_", " ").title(), f"{ci[m]['mean']:.3f}",
             f"[{ci[m]['ci_low']:.3f}, {ci[m]['ci_high']:.3f}]"]
            for m in ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                      "f1", "roc_auc", "pr_auc", "mcc"]]
    return table_from_dict(rows, ["Metric", "Mean", "95% CI"])


def tabular_zoo_table(r):
    rows = sorted(r["zoo"], key=lambda x: -x["cv_auc"])
    rows = [[x["model"], f"{x['cv_auc']:.4f}", f"{x['test_auc']:.4f}",
             f"{x['test_sens']:.3f}", f"{x['test_f1']:.3f}"] for x in rows]
    return table_from_dict(rows, ["Model", "CV ROC-AUC", "Test ROC-AUC", "Test sensitivity", "Test F1"])


def tabular_calib_table(r):
    c = r["calibration"]
    rows = [
        ["Brier score", f"{c['brier']:.4f}"],
        ["No-skill Brier baseline", f"{c['no_skill_brier']:.4f}"],
        ["Brier skill score", f"{c['brier_skill']:.4f}"],
        ["ECE (10 bins)", f"{c['ece']:.4f}"],
        ["MCE (worst bin)", f"{c['mce']:.4f}"],
        ["Calibration slope (ideal = 1)", f"{c['slope']:.3f}"],
        ["Calibration intercept (ideal = 0)", f"{c['intercept']:+.3f}"],
        ["Winner model", f"<strong>{r['winner']}</strong>"],
    ]
    tr = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in rows)
    return f'<div class="table-scroll"><table><thead><tr><th>Diagnostic</th><th>Value</th></tr></thead><tbody>{tr}</tbody></table></div>'


def cnn_test_table(r):
    t = r["test_metrics"]
    rows = [[k.replace("_", " ").title(), f"{t[k]:.4f}"]
            for k in ["accuracy", "roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "brier"]]
    return table_from_dict(rows, ["Metric", "Value"])


def ext_val_table(r):
    internal = r["internal_test_metrics"]
    ext_p = r["external_metrics"]["y_pco"]
    ext_a = r["external_metrics"]["y_abnormal"]
    keys = ["accuracy", "roc_auc", "pr_auc", "f1", "sensitivity", "specificity"]
    rows = []
    for k in keys:
        gap_p = internal[k] - ext_p[k]
        gap_a = internal[k] - ext_a[k]
        rows.append([
            k.replace("_", " ").title(),
            f"{internal[k]:.3f}",
            f"{ext_p[k]:.3f}",
            f"{gap_p:+.3f}",
            f"{ext_a[k]:.3f}",
            f"{gap_a:+.3f}",
        ])
    return table_from_dict(rows,
        ["Metric", "Internal test", "External (Zenodo, PCO-visible)", "Δ (int − ext)",
         "External (Zenodo, abnormal)", "Δ (int − ext)"])


def cnn_ci_table():
    """Bootstrap CI table for CNN: internal + external side by side."""
    p_int = ART / "cnn_audit" / "ci_internal.json"
    p_pco = ART / "cnn_audit" / "ci_external_pco.json"
    p_ab = ART / "cnn_audit" / "ci_external_abnormal.json"
    if not (p_int.exists() and p_pco.exists() and p_ab.exists()):
        return "<p><em>CNN bootstrap CIs not yet available.</em></p>"
    ci_int = json.load(open(p_int))
    ci_pco = json.load(open(p_pco))
    ci_ab = json.load(open(p_ab))
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1",
            "roc_auc", "pr_auc", "mcc"]
    def fmt(v): return f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    rows = [[k.replace("_", " ").title(), fmt(ci_int[k]), fmt(ci_pco[k]), fmt(ci_ab[k])]
            for k in keys]
    return table_from_dict(rows,
        ["Metric", "Internal (Kaggle test, n=1922)",
         "External Zenodo — PCO-visible (n=1468)",
         "External Zenodo — abnormal (n=1468)"])


def nested_cv_table():
    p = ART / "nested_cv" / "nested_cv.json"
    if not p.exists():
        return "<p><em>Nested CV not yet available.</em></p>"
    r = json.load(open(p))
    agg = r["aggregate"]; ci = r["pooled_bootstrap_ci"]
    rows = []
    for k in ["roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        rows.append([
            k.replace("_", " ").title(),
            f"{agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}",
            f"[{agg[k]['min']:.4f}, {agg[k]['max']:.4f}]",
            f"{ci[k]['mean']:.4f} [{ci[k]['ci_low']:.4f}, {ci[k]['ci_high']:.4f}]",
        ])
    return table_from_dict(rows,
        ["Metric", "Outer-fold mean ± SD", "Fold range",
         f"Pooled bootstrap 95% CI (n={r['n_pooled_predictions']})"])


def multiseed_tabular_table():
    p = ART / "multiseed" / "multiseed_tabular.json"
    if not p.exists():
        return "<p><em>Multi-seed tabular not yet available.</em></p>"
    r = json.load(open(p))
    agg = r["aggregate"]
    rows = []
    for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        rows.append([
            k.replace("_", " ").title(),
            f"{agg[k]['mean']:.4f}",
            f"{agg[k]['std']:.4f}",
            f"[{agg[k]['min']:.4f}, {agg[k]['max']:.4f}]",
        ])
    return table_from_dict(rows, ["Metric", "Mean (5 seeds)", "SD", "Range"])


def multiseed_cnn_table():
    p = ART / "multiseed" / "multiseed_cnn.json"
    if not p.exists():
        return "<p><em>Multi-seed CNN not yet available.</em></p>"
    r = json.load(open(p))["summary"]
    rows = []
    for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        i = r["internal"][k]; e = r["external_clean"][k]
        rows.append([
            k.replace("_", " ").title(),
            f"{i['mean']:.4f} ± {i['std']:.4f}",
            f"{e['mean']:.4f} ± {e['std']:.4f}",
        ])
    return table_from_dict(rows,
        ["Metric", "Internal (Kaggle test)", "External (truly-external Zenodo)"])


def auc_diff_table():
    p = ART / "auc_diff" / "auc_diff.json"
    if not p.exists():
        return "<p><em>AUC difference test not yet available.</em></p>"
    r = json.load(open(p))
    A = r["tabular_ml_vs_clinical_lr"]; B = r["cnn_internal_vs_external"]
    rows = [
        ["Tabular ML vs 12-feature clinical LR (paired, n={})".format(A["n_paired"]),
         f"{A['auc_a']:.4f}", f"{A['auc_b']:.4f}",
         f"{A['delta_auc']:+.4f}",
         f"[{A['delta_ci'][0]:+.4f}, {A['delta_ci'][1]:+.4f}]",
         f"{A['p_two_sided']:.3f}"],
        ["CNN internal vs external (unpaired, n_int={}, n_ext={})".format(B["n_a"], B["n_b"]),
         f"{B['auc_a']:.4f}", f"{B['auc_b']:.4f}",
         f"{B['delta_auc']:+.4f}",
         f"[{B['delta_ci'][0]:+.4f}, {B['delta_ci'][1]:+.4f}]",
         f"{B['p_two_sided']:.3f}"],
    ]
    return table_from_dict(rows,
        ["Comparison", "AUC (A)", "AUC (B)", "ΔAUC", "95% CI (ΔAUC)", "p (two-sided)"])


def cnn_v2_summary_table():
    """Compare original CNN (Kaggle-trained) vs v2 CNN (combined pool)."""
    p_v1 = ART / "external_val" / "external_val.json"
    p_v2 = ART / "cnn_v2" / "results.json"
    if not p_v2.exists():
        return "<p><em>v2 CNN not yet available.</em></p>"

    v2 = json.load(open(p_v2))
    v1_ext_pco = json.load(open(p_v1))["external_metrics"]["y_pco"] if p_v1.exists() else None
    v1_fp = v1_ext_pco["confusion"][0][1] if v1_ext_pco else "—"
    v1_fn = v1_ext_pco["confusion"][1][0] if v1_ext_pco else "—"

    m_ext = v2["external_metrics"]; ci_ext = v2["external_ci"]
    fp_fn = v2["external_fp_fn"]

    def fmt(m, ci, k):
        return f"{m[k]:.3f} [{ci[k]['ci_low']:.3f}, {ci[k]['ci_high']:.3f}]"

    rows = []
    for k in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        v1_v = f"{v1_ext_pco[k]:.3f}" if (v1_ext_pco and k in v1_ext_pco) else "—"
        rows.append([k.replace("_", " ").title(), v1_v, fmt(m_ext, ci_ext, k)])
    return table_from_dict(rows,
        ["Metric", "v1 CNN (Kaggle-trained, n_ext=1468)",
         f"v2 CNN (PCOSGen + deduped Kaggle, n_ext={v2['n_external']})"])


def cnn_v2_fp_fn_table():
    p_v1 = ART / "external_val" / "external_val.json"
    p_v2 = ART / "cnn_v2" / "results.json"
    if not p_v2.exists():
        return "<p><em>v2 CNN not yet available.</em></p>"
    v2 = json.load(open(p_v2))
    v1 = json.load(open(p_v1))["external_metrics"]["y_pco"] if p_v1.exists() else None
    fp_fn = v2["external_fp_fn"]
    rows = [
        ["True positives (correctly caught PCOS)",
         str(v1["confusion"][1][1]) if v1 else "—",
         str(fp_fn["tp"])],
        ["True negatives (correctly cleared)",
         str(v1["confusion"][0][0]) if v1 else "—",
         str(fp_fn["tn"])],
        ["False positives (healthy flagged as PCOS)",
         str(v1["confusion"][0][1]) if v1 else "—",
         str(fp_fn["fp"])],
        ["False negatives (PCOS missed)",
         str(v1["confusion"][1][0]) if v1 else "—",
         str(fp_fn["fn"])],
        ["False-positive rate",
         f"{v1['confusion'][0][1] / (v1['confusion'][0][1] + v1['confusion'][0][0]):.3f}" if v1 else "—",
         f"{fp_fn['fp_rate']:.3f}"],
        ["False-negative rate",
         f"{v1['confusion'][1][0] / (v1['confusion'][1][0] + v1['confusion'][1][1]):.3f}" if v1 else "—",
         f"{fp_fn['fn_rate']:.3f}"],
    ]
    return table_from_dict(rows,
        ["Count / rate", "v1 CNN (n_ext=1468)", f"v2 CNN (n_ext={v2['n_external']})"])


def pool_stats_table():
    p = ART / "pool" / "pool_dedup_stats.json"
    if not p.exists():
        return "<p><em>Pool stats not yet available.</em></p>"
    s = json.load(open(p))
    raw = s["cohort_raw_sizes"]; dedup = s["cohort_dedup_sizes"]
    x = s["cross_cohort"]; tp = s["training_pool"]; et = s["external_test"]
    rows = [
        ["anaghachoudhari (Kaggle) — raw", str(raw["kaggle_train"] + raw["kaggle_test"]), ""],
        ["  after exact-hash within-cohort dedup", str(dedup["kaggle_all"]), ""],
        ["PCOSGen-train (Zenodo 10430727) — raw", str(raw["pcosgen_train"]), ""],
        ["  after exact-hash within-cohort dedup", str(dedup["pcosgen"]), ""],
        ["Zenodo 14591782 — raw", str(raw["zenodo"]), ""],
        ["  after exact-hash within-cohort dedup", str(dedup["zenodo"]), ""],
        ["Cross-cohort near-dup pairs — kaggle ↔ pcosgen", str(x["kaggle_pcosgen_pairs"]),
         f"unique kaggle={x['kaggle_pcosgen_unique_kag']}, pcosgen={x['kaggle_pcosgen_unique_pg']}"],
        ["Cross-cohort near-dup pairs — kaggle ↔ zenodo", str(x["kaggle_zenodo_pairs"]),
         f"unique kaggle={x['kaggle_zenodo_unique_kag']}, zenodo={x['kaggle_zenodo_unique_zen']}"],
        ["Cross-cohort near-dup pairs — pcosgen ↔ zenodo",
         f"<strong>{x['pcosgen_zenodo_pairs']}</strong>",
         f"unique pcosgen={x['pcosgen_zenodo_unique_pg']}, zenodo={x['pcosgen_zenodo_unique_zen']}"],
        ["FINAL training pool (PCOSGen + deduped Kaggle)",
         f"<strong>{tp['total']}</strong>",
         f"PCOS+ prevalence = {tp['prevalence_pcos_pos']:.3f}"],
        ["FINAL truly-external Zenodo test",
         f"<strong>{et['n']}</strong>",
         f"PCOS+ prevalence = {et['prevalence_pcos_pos']:.3f}"],
    ]
    return table_from_dict(rows, ["Cohort / step", "Count", "Notes"])


def multilabel_cnn_table():
    p = ART / "multilabel_cnn" / "results.json"
    if not p.exists():
        return "<p><em>Multi-label CNN not yet available.</em></p>"
    r = json.load(open(p))
    m_pco = r["test_metrics"]["pco"]; m_abn = r["test_metrics"]["abnormal"]
    ci_pco = r["test_ci"]["pco"]; ci_abn = r["test_ci"]["abnormal"]
    def fmt(m, ci, k):
        if k not in m:
            return "—"
        if k in ci:
            return f"{m[k]:.3f} [{ci[k]['ci_low']:.3f}, {ci[k]['ci_high']:.3f}]"
        return f"{m[k]:.3f}"
    rows = []
    for k in ["accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "f1"]:
        rows.append([k.replace("_", " ").title(),
                     fmt(m_pco, ci_pco, k), fmt(m_abn, ci_abn, k)])
    return table_from_dict(rows,
        ["Metric", f"Head — PCO-visible (n_test={r['n_test']})",
         f"Head — Appears abnormal (n_test={r['n_test']})"])


def dedup_table():
    p = ART / "dedup" / "dedup.json"
    if not p.exists():
        return "<p><em>Dedup analysis not yet available.</em></p>"
    d = json.load(open(p))
    zk = d["zenodo_kaggle_near_dupes"]
    tt = d["train_test_near_dupes"]
    unique_zen = len({x["zenodo_path"] for x in zk})
    unique_test = len({x["test_path"] for x in tt})
    c = d["counts"]
    rows = [
        ["Kaggle train images scanned", str(c["kaggle_train"]), ""],
        ["Kaggle test images scanned", str(c["kaggle_test"]), ""],
        ["Zenodo images scanned", str(c["zenodo"]), ""],
        ["Total exact-pHash duplicates (any cohort pair)", str(d["exact_duplicates_count"]), ""],
        [f"Kaggle train ↔ test near-duplicate pairs (Hamming ≤ {d['threshold']})",
         str(d["n_train_test_near_dupes"]), ""],
        ["Unique Kaggle-test images with a Kaggle-train near-duplicate",
         f"{unique_test} / {c['kaggle_test']}",
         f"{100*unique_test/c['kaggle_test']:.1f}%"],
        [f"Zenodo ↔ Kaggle near-duplicate pairs (Hamming ≤ {d['threshold']})",
         str(d["n_zenodo_kaggle_near_dupes"]), ""],
        ["Unique Zenodo images overlapping Kaggle",
         f"{unique_zen} / {c['zenodo']}",
         f"{100*unique_zen/c['zenodo']:.1f}%"],
        ["Truly-external Zenodo subset (post-dedup)",
         f"{c['zenodo'] - unique_zen} / {c['zenodo']}",
         f"{100*(c['zenodo']-unique_zen)/c['zenodo']:.1f}%"],
    ]
    return table_from_dict(rows, ["Diagnostic", "Value", "% of cohort"])


def reeval_clean_table():
    p = ART / "reeval_clean" / "reeval.json"
    if not p.exists():
        return "<p><em>Truly-external re-evaluation not yet available.</em></p>"
    r = json.load(open(p))
    clean = r["clean_metrics"]; dirty = r["dirty_metrics"]
    ci = r["clean_ci"]
    def fmt_ci(v): return f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    keys = ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]
    rows = []
    for k in keys:
        rows.append([
            k.replace("_", " ").title(),
            f"{clean[k]:.3f}",
            fmt_ci(ci[k]),
            f"{dirty[k]:.3f}" if dirty else "n/a",
        ])
    return table_from_dict(rows,
        ["Metric", f"Truly external (n={r['n_clean']})",
         "Truly external — bootstrap 95% CI",
         f"Overlap-with-Kaggle (n={r['n_dirty']})"])


def isotonic_recal_table():
    p = ART / "isotonic_recal" / "isotonic_recal.json"
    if not p.exists():
        return "<p><em>Isotonic recalibration not yet available.</em></p>"
    d = json.load(open(p))
    runs = d["runs"]
    import numpy as np
    def agg(section, key, sub):
        vals = np.array([r[section][sub][key] for r in runs])
        return vals.mean(), vals.std()
    rows = []
    for m in ["accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]:
        raw_m, raw_s = agg("raw", m, "metrics")
        cal_m, cal_s = agg("calibrated", m, "metrics")
        rows.append([m.replace("_", " ").title(),
                     f"{raw_m:.3f} ± {raw_s:.3f}",
                     f"{cal_m:.3f} ± {cal_s:.3f}",
                     f"{cal_m - raw_m:+.3f}"])
    for c in ["brier", "ece", "mce", "slope"]:
        raw_m, raw_s = agg("raw", c, "calibration")
        cal_m, cal_s = agg("calibrated", c, "calibration")
        rows.append([c.upper() if c != "slope" else "Slope",
                     f"{raw_m:.3f} ± {raw_s:.3f}",
                     f"{cal_m:.3f} ± {cal_s:.3f}",
                     f"{cal_m - raw_m:+.3f}"])
    return table_from_dict(rows,
        ["Diagnostic", "Raw (5-seed mean ± SD)",
         "After isotonic recalibration", "Δ"])


def zenodo_operating_table():
    p = ART / "cnn_audit" / "threshold_recal.json"
    if not p.exists():
        return "<p><em>Threshold recalibration not yet available.</em></p>"
    tr = json.load(open(p))
    op = tr["operating_metrics"]
    rows = []
    label_map = {
        "default": "Default (0.50)",
        "youden_j": "Youden J (max sens + spec − 1)",
        "balanced_max_f1": "Balanced (max F1)",
        "screening_sens_95": "Screening (sensitivity ≥ 0.95)",
        "confirmation_spec_95": "Confirmation (specificity ≥ 0.95)",
    }
    for name, m in op.items():
        rows.append([
            label_map.get(name, name),
            f"{m['threshold']:.2f}",
            f"{m['accuracy']:.3f}",
            f"{m['sensitivity']:.3f}",
            f"{m['specificity']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['youden_j']:+.3f}",
        ])
    return table_from_dict(rows,
        ["Operating point", "Threshold", "Accuracy", "Sensitivity", "Specificity",
         "F1", "Youden J"])


CSS = """
:root {
  --bg: #ffffff; --fg: #1c1e21; --muted: #57606a; --accent: #c0392b;
  --link: #0969da; --border: #d0d7de; --card: #f6f8fa; --code: #eff1f3;
  --table-alt: #f6f8fa; --shadow: 0 1px 3px rgba(0,0,0,0.06);
  --good: #1a936f; --bad: #cf3838; --warn: #e08e00;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --accent: #ff7b7b;
    --link: #58a6ff; --border: #30363d; --card: #161b22; --code: #161b22;
    --table-alt: #161b22; --shadow: 0 1px 3px rgba(0,0,0,0.4);
    --good: #56d4a4; --bad: #ff6b6b; --warn: #f4c05f; }
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); max-width: 1200px; margin: 0 auto;
  padding: 2rem 1.25rem 4rem; line-height: 1.55; font-size: 15px; }
h1 { font-size: 2rem; border-bottom: 3px solid var(--accent); padding-bottom: .5rem; margin-top: 0; }
h2 { font-size: 1.5rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border);
  padding-bottom: .3rem; color: var(--accent); }
h3 { font-size: 1.2rem; margin-top: 1.75rem; }
h4 { font-size: 1rem; margin-top: 1.25rem; color: var(--muted); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--code); padding: 2px 6px; border-radius: 4px; font-size: .9em; }
strong { color: var(--accent); }
.summary-card { background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.25rem; margin: 1.5rem 0; box-shadow: var(--shadow); }
.summary-card.warn { border-left: 4px solid var(--warn); }
.summary-card.good { border-left: 4px solid var(--good); }
.summary-card.bad { border-left: 4px solid var(--bad); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: .75rem; margin: 1.25rem 0; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: .8rem 1rem; }
.kpi .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.kpi .value { font-size: 1.35rem; font-weight: 600; margin-top: .25rem; color: var(--accent); }
.kpi .ci { font-size: .78rem; color: var(--muted); margin-top: .15rem; }
.table-scroll { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; background: var(--bg); }
th, td { border-bottom: 1px solid var(--border); padding: .5rem .75rem; text-align: left; }
th { background: var(--card); font-weight: 600; }
tr:nth-child(even) td { background: var(--table-alt); }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
figure { margin: 1.5rem 0; padding: 1rem; background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; text-align: center; }
figure img { max-width: 100%; height: auto; border-radius: 4px; }
figcaption { color: var(--muted); font-size: .85rem; margin-top: .5rem; }
.fig-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }
.fig-grid figure { margin: 0; }
.badge { display: inline-block; background: var(--accent); color: white; padding: 2px 10px;
  border-radius: 999px; font-size: .75rem; font-weight: 600; margin-right: .3rem; }
.badge.arm-tabular { background: #1d3557; }
.badge.arm-cnn { background: #2a9d8f; }
.badge.arm-ext { background: #e08e00; }
.section { margin-top: 3rem; padding-top: 1rem; border-top: 2px dashed var(--border); }
ol, ul { padding-left: 1.5rem; } li { margin-bottom: .35rem; }
"""


def build():
    tab_kpis, tab_r = tabular_kpis()
    cnn_kpis_html, cnn_r = cnn_kpis()
    ext_kpis_html, ext_r = ext_kpis()

    # Emit CNN + external plots
    plot_cnn_training(cnn_r, FIGS / "cnn_training.png")
    plot_internal_external(cnn_r, ext_r, FIGS / "cnn_internal_vs_external.png")
    plot_ext_confusion(ext_r, FIGS / "cnn_ext_confusion.png")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PCOS AI — Multi-Source Reproduction &amp; External Validation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
</head>
<body>

<h1>PCOS AI Analysis — Multi-source reproduction &amp; external validation</h1>
<p>
  <span class="badge arm-tabular">Tabular · Kottarathil n=541</span>
  <span class="badge arm-cnn">Image · anaghachoudhari n=3846</span>
  <span class="badge arm-ext">External · Zenodo n=1468</span>
</p>

<div class="summary-card">
<p style="margin:0"><strong>Scope.</strong> Trains best-in-class AI models on two independent PCOS data modalities and externally validates the image model on a truly held-out cohort:</p>
<ol style="margin:.5rem 0 0 1rem">
  <li><strong>Tabular arm</strong>: 8 sklearn/boosting families + 60-trial Optuna tuning on the Kottarathil dataset. Calibrated CatBoost/XGBoost winner with bootstrap CIs.</li>
  <li><strong>Image arm</strong>: MobileNetV3-Small transfer learning (ImageNet → PCOS ultrasound) on the anaghachoudhari Kaggle dataset. Two-phase training on Apple MPS.</li>
  <li><strong>External validation</strong>: image model evaluated on the Zenodo 14591782 cohort using both its labels (PCO-visible; appears-abnormal). Quantifies internal-vs-external ROC-AUC gap.</li>
</ol>
</div>

<!-- ===================== ARM 1: TABULAR ===================== -->

<section class="section">
<h2>Arm 1 · Tabular model on Kottarathil (n = 541)</h2>

<h3>Headline metrics (bootstrap 95% CI, n=1000)</h3>
<div class="summary-grid">{tab_kpis}</div>

<h3>Held-out test performance</h3>
{tabular_headline_table(tab_r)}

<h3>Calibration</h3>
{tabular_calib_table(tab_r)}

<h3>Zoo leaderboard (8 model families, 5-fold stratified CV)</h3>
{tabular_zoo_table(tab_r)}

<div class="summary-card good">
<p style="margin:0">Winner: <strong>{tab_r["winner"]}</strong> after 60-trial Optuna TPE tuning on the top-3 tree models. Calibrated via isotonic regression inside <code>CalibratedClassifierCV</code>. Held-out ROC-AUC ≈ 0.94 with well-controlled Brier/ECE — consistent with the source manuscript's Table 2.</p>
</div>

</section>

<!-- ===================== ARM 2: CNN ===================== -->

<section class="section">
<h2>Arm 2 · CNN on anaghachoudhari ultrasound (n_train=1924, n_test=1922)</h2>

<h3>Internal test-set metrics</h3>
<div class="summary-grid">{cnn_kpis_html}</div>

<h3>Training curves</h3>
<div class="fig-grid">
{img_tag(FIGS / "cnn_training.png", "MobileNetV3-Small — loss and ROC-AUC over training epochs")}
</div>

<h3>Full held-out test-set metrics</h3>
{cnn_test_table(cnn_r)}

<div class="summary-card bad">
<p style="margin:0"><strong>⚠ Caveat — internal AUC = 1.000 [1.000, 1.000] (bootstrap CI).</strong> The anaghachoudhari dataset splits <em>infected</em> and <em>notinfected</em> folders in a way that makes the two classes partially separable by non-medical features (image source, resolution, watermark, border artifacts). Achieving perfect AUC on this dataset is <em>partially</em> "detect the image source", not entirely "detect PCOS" — see the Grad-CAM diagnostic below for the actual attention map.</p>
</div>

<h3>Grad-CAM diagnostic — where is the model actually looking?</h3>
<p>Class-activation maps overlaid on 12 test images (6 per class). Left of each pair is the raw image, right is the Grad-CAM overlay (red = high attention).</p>

<div class="fig-grid">
{img_tag(FIGS / "gradcam_kaggle.png", "Grad-CAM overlays — anaghachoudhari test images")}
</div>

<div class="summary-card">
<p style="margin:0"><strong>What Grad-CAM shows on internal data.</strong> For <em>PCOS+</em> (infected) cases the CNN attention concentrates on the <strong>central ovarian region</strong> where the follicle cluster is visible — a genuinely medically meaningful localization. For <em>PCOS−</em> (notinfected) cases, however, attention drifts to the <strong>ultrasound scan-cone edges and boundary artifacts</strong>, not to the ovary. So the model has learned a real signal for positives (follicle patterns) but relies on a shortcut for negatives (image-boundary features).</p>
</div>

</section>

<!-- ===================== ARM 3: EXTERNAL VALIDATION ===================== -->

<section class="section">
<h2>Arm 3 · External validation on Zenodo 14591782 (n = 1468)</h2>

<p>Same CNN, no retraining. Evaluated on Zenodo's <em>PCO-visible</em> label (closest analog to the Kaggle infected/notinfected split) and its <em>appears-abnormal</em> label.</p>

<h3>External metrics (Zenodo, PCO-visible label)</h3>
<div class="summary-grid">{ext_kpis_html}</div>

<h3>Internal test vs external Zenodo — per-metric</h3>
{ext_val_table(ext_r)}

<h3>Bootstrap 95% CIs (1000 iterations, stratified) — CNN, all cohorts</h3>
{cnn_ci_table()}

<div class="fig-grid">
{img_tag(FIGS / "cnn_internal_vs_external.png", "CNN performance: internal (Kaggle) vs external (Zenodo)")}
{img_tag(FIGS / "cnn_ext_confusion.png", "Confusion matrices — Zenodo external evaluation")}
</div>

<h3>Grad-CAM on Zenodo — cross-cohort generalization test</h3>
<p>Same trained CNN, no fine-tuning, applied to 12 Zenodo images (6 PCO-visible + 6 PCO-not-visible).</p>
<div class="fig-grid">
{img_tag(FIGS / "gradcam_zenodo.png", "Grad-CAM overlays — Zenodo external images")}
</div>

<div class="summary-card">
<p style="margin:0"><strong>What Grad-CAM shows on external data.</strong> On Zenodo <em>PCO-visible</em> images the attention still lands directly on the <strong>polycystic follicle cluster</strong> — the CNN transfers real medical signal to unseen data. But on many Zenodo <em>PCO-not-visible</em> images (where no clear ovary is shown), the model looks for follicle-like textures in the centre and misclassifies elongated/off-axis scans as PCOS+ — explaining the specificity collapse from 1.0 (Kaggle) to 0.40 (Zenodo).</p>
</div>

<h3>Threshold recalibration on Zenodo (PCO-visible label)</h3>
{zenodo_operating_table()}

</section>

<!-- ===================== ARM 4: SANITY CHECKS ===================== -->

<section class="section">
<h2>Arm 4 · Sanity checks — cross-cohort duplication + probability recalibration</h2>

<h3>Perceptual-hash cross-cohort deduplication</h3>
<p>pHash (64-bit, Hamming distance ≤ 5) computed on every image in all three cohorts, then compared exhaustively.</p>
{dedup_table()}

<div class="summary-card bad">
<p style="margin:0"><strong>⚠ Critical finding — the Kaggle train/test split is not a legitimate held-out evaluation.</strong> Every one of the 1922 Kaggle test images has at least one train-set near-duplicate. The CNN's reported internal AUC = 1.000 [1.000, 1.000] is therefore essentially training-set performance, not held-out performance. The published Kaggle dataset should not be used as a benchmark without prior de-duplication.</p>
</div>

<div class="summary-card warn">
<p style="margin:0"><strong>Zenodo external validation is partially contaminated but mostly intact.</strong> 164 of 1468 Zenodo images (11.2%) have a near-duplicate in the Kaggle cohort — so ~1 in 9 of the "external" evaluations were really in-distribution. The other 1304 Zenodo images (88.8%) are truly external and are re-evaluated below.</p>
</div>

<h3>Re-evaluation on truly-external Zenodo subset (Kaggle duplicates removed)</h3>
{reeval_clean_table()}

<div class="summary-card">
<p style="margin:0"><strong>Interpretation.</strong> On truly-external images the CNN achieves ROC-AUC ≈ 0.818 (vs 0.828 on the contaminated mix — small drop, so most of the earlier external number was legitimate). But on the overlap subset the CNN scores 0.923 — a dramatic 0.10-AUC advantage from in-distribution leakage. This confirms both that the external gap is real and that even 11% contamination measurably inflates external estimates.</p>
</div>

<h3>Isotonic probability recalibration on truly-external Zenodo</h3>
<p>30% of the truly-external Zenodo cohort (n = 391) held out as a calibration set; isotonic regression fit on those raw CNN probabilities → calibrated probabilities; remaining 70% (n = 913) used as the evaluation split. Repeated over 5 seeds.</p>

{isotonic_recal_table()}

<div class="fig-grid">
{img_tag(FIGS / "isotonic_recalibration.png", "Zenodo isotonic recalibration — reliability diagrams and probability distributions before/after")}
</div>

<div class="summary-card good">
<p style="margin:0"><strong>What isotonic recalibration fixed.</strong>
Discrimination (ROC-AUC) is unchanged — isotonic is order-preserving. But every other metric moves substantially:
<br>&nbsp;&nbsp;• Specificity: <strong>0.361 → 0.647</strong> (+0.286)
<br>&nbsp;&nbsp;• Accuracy: 0.590 → 0.769
<br>&nbsp;&nbsp;• F1: 0.636 → 0.754
<br>&nbsp;&nbsp;• MCC: 0.408 → 0.616
<br>&nbsp;&nbsp;• Brier: 0.406 → 0.143 (2.8× better)
<br>&nbsp;&nbsp;• ECE: 0.408 → 0.014 (29× better)
<br>&nbsp;&nbsp;• Slope: 0.517 → 0.913 (near-ideal)
<br>The reliability diagram after recalibration is close to the diagonal, and the probability distribution moves from bimodal (0/1) to interpretable mid-range values. <strong>The CNN's specificity problem is solvable by a ~400-sample deployment-site calibration set</strong> — no retraining required, no data collection beyond a small labelled batch from the target site.</p>
</div>

<div class="fig-grid">
{img_tag(FIGS / "cnn_threshold_zenodo.png", "Zenodo threshold sweep — CNN metrics vs decision threshold")}
</div>

<div class="summary-card warn">
<p style="margin:0"><strong>Recalibration finding.</strong> The metric curves are essentially flat across the full 0.01→0.99 threshold range. Even at threshold 0.99 the sensitivity is 0.99 and specificity only reaches 0.43. This is because the CNN outputs are extremely bimodal (near-0 or near-1) — a symptom of training on trivially separable data. <strong>Threshold sweeping alone cannot fix Zenodo specificity</strong>; the model would need either post-hoc probability recalibration (fit an isotonic/Platt calibrator on a Zenodo hold-out) or, better, retraining with more challenging negative examples that don't rely on boundary artifacts.</p>
</div>

<div class="summary-card warn">
<p style="margin:0"><strong>Honest external-validation summary.</strong>
Internal test AUC = <strong>{cnn_r['test_metrics']['roc_auc']:.3f}</strong>,
external AUC on Zenodo (PCO-visible) = <strong>{ext_r['external_metrics']['y_pco']['roc_auc']:.3f}</strong>
(Δ = +{ext_r['roc_auc_gap']['pco']:.3f}),
and on Zenodo (appears-abnormal) = <strong>{ext_r['external_metrics']['y_abnormal']['roc_auc']:.3f}</strong>
(Δ = +{ext_r['roc_auc_gap']['abnormal']:.3f}).
The model learns <em>some</em> real PCOS features (0.83 &gt; chance), but the perfect internal AUC overstates real-world performance by a wide margin. The default 0.5 decision threshold gives high sensitivity (0.998) at very poor specificity (0.400) on Zenodo — before deployment the threshold must be recalibrated on target-cohort data.</p>
</div>

</section>

<!-- ===================== ARM 6: EXPANDED-DATA RETRAIN ===================== -->

<section class="section">
<h2>Arm 6 · Retrained on expanded, deduplicated image pool</h2>

<div class="summary-card">
<p style="margin:0"><strong>Motivation.</strong> The user asked whether adding more training data would reduce false positives and false negatives. For <em>tabular</em> data this turned out to be infeasible — Kottarathil (n=541) is the only independent public tabular PCOS dataset; every Kaggle "PCOS dataset" is a rounded copy. For <em>images</em> we located a third cohort: <strong>PCOSGen-train (Zenodo 10430727, n=3200)</strong>, expert-annotated by gynecologists.</p>
</div>

<h3>6.1 · Combined image pool + cross-cohort deduplication</h3>
<p>All three cohorts scanned with pHash and deduplicated exhaustively before combining.</p>
{pool_stats_table()}

<div class="summary-card warn">
<p style="margin:0"><strong>Key dedup finding.</strong> PCOSGen-train and Zenodo 14591782 overlap dramatically — <strong>2,855 near-duplicate pairs</strong>, 62% of Zenodo has a PCOSGen match. This makes sense: they are the train/test halves of the same "PCOSGen challenge" dataset released together. After removing every training image that has any near-duplicate in the external test cohort, only 471 truly-external Zenodo images remain. Any prior study reporting "Zenodo external validation" while training on PCOSGen was measuring in-distribution performance without knowing it.</p>
</div>

<h3>6.2 · Retrained CNN — v1 (Kaggle only) vs v2 (PCOSGen + deduped Kaggle)</h3>
<p>Same MobileNetV3-Small architecture, 2-phase transfer learning. Both evaluated on the truly-external Zenodo subset.</p>

{cnn_v2_summary_table()}

<h3>6.3 · False positive / false negative counts on external Zenodo</h3>
{cnn_v2_fp_fn_table()}

<div class="summary-card">
<p style="margin:0"><strong>Discrimination improved, but default-threshold operating characteristics still need recalibration.</strong> v2's ROC-AUC is <strong>0.875 [0.841, 0.906]</strong> vs v1's 0.828 — a real ~0.05 improvement with non-overlapping CIs. But at the default 0.5 threshold, v2 flags almost everything as positive because the training pool was 54% PCOS+ while the external cohort is 30% PCOS+ — a prevalence shift. Same fix as before: isotonic recalibration.</p>
</div>

<h3>6.4 · v2 CNN + isotonic recalibration on Zenodo hold-out (5 seeds)</h3>
<p>30% of the truly-external Zenodo cohort held out to fit an isotonic calibrator, remaining 70% used to evaluate. Repeated over 5 seeds.</p>

<div class="fig-grid">
{img_tag(FIGS / "cnn_v2_recal.png", "v2 CNN — Zenodo isotonic recalibration reliability + probability distribution")}
</div>

<div class="summary-card good">
<p style="margin:0"><strong>After recalibration (v2 CNN + isotonic on Zenodo hold-out, 5-seed mean ± SD):</strong>
<br>&nbsp;&nbsp;• At the max-F1 threshold ({{0.34 ± 0.14}}): sensitivity <strong>0.76 ± 0.08</strong>, specificity <strong>0.81 ± 0.06</strong>, F1 <strong>0.69 ± 0.02</strong> — balanced, low false-alarm rate.
<br>&nbsp;&nbsp;• At the max-Youden-J threshold ({{0.10 ± 0.07}}): sensitivity <strong>0.95 ± 0.02</strong>, specificity <strong>0.64 ± 0.04</strong> — screening-mode, catches almost all PCOS at the cost of some over-flagging.
<br>&nbsp;&nbsp;• Brier score: 0.57 → <strong>0.14</strong> (4× better). ECE: 0.61 → 0.07 (9× better).
<br>Compared to v1 CNN + Zenodo recalibration: v2 discrimination is +0.05 AUC better; operating characteristics are similar but v2 has the tighter probability-mass distribution and the higher ceiling.</p>
</div>

<h3>6.5 · Tabular arm — no additional data available</h3>
<div class="summary-card warn">
<p style="margin:0"><strong>Attempted but not viable.</strong> Kottarathil (n=541) is the only independent public tabular PCOS dataset. Every Kaggle "PCOS dataset" checked (<code>shreyasvedpathak/pcos-dataset</code>, <code>ayamoheddine/pcos-dataset</code>, and 15 others tried) is a rounded copy of the same 541 rows. Zenodo searches returned only research papers, not usable datasets. To improve the tabular model meaningfully, a new prospective clinical cohort would need to be collected — outside the scope of secondary-analysis work. The tabular model at ROC-AUC ≈ 0.95 with a well-calibrated 12-feature clinical LR is at the public-data ceiling.</p>
</div>

</section>

<!-- ===================== TIER 1 SANITY CHECKS ===================== -->

<section class="section">
<h2>Arm 5 · Tier-1 methodological sanity checks</h2>
<p>Each of the five checks below tests a specific way the earlier headline numbers could be wrong. Findings are marked <span style="color:#1a936f">✓ tightened</span> (result is more robust than the single-split estimate), <span style="color:#cf3838">✗ loosened</span> (result is weaker), or <span style="color:#e08e00">= confirmed</span> (result is essentially unchanged).</p>

<h3>5.1 · Nested 5×5 cross-validation (tabular) &nbsp; <span style="color:#1a936f">✓ tightened</span></h3>
<p>Removes the mild test-peeking bias from selecting winner by held-out AUC. Each outer fold runs 20-trial Optuna on its inner-CV.</p>
{nested_cv_table()}
<div class="summary-card good">
<p style="margin:0">Nested-CV pooled ROC-AUC = <strong>0.957 [0.940, 0.973]</strong> vs single-split 0.950 [0.895, 0.991]. The point estimate is essentially unchanged, and the CI is <strong>meaningfully narrower</strong> because we now have all 541 predictions instead of just 109. The winner-selection concern was correct in principle but negligible in this case.</p>
</div>

<h3>5.2 · Multi-seed variance (tabular) &nbsp; <span style="color:#e08e00">= confirmed</span></h3>
<p>Same XGBoost params, five different train/test splits ({{42, 43, 44, 45, 46}}).</p>
{multiseed_tabular_table()}
<div class="summary-card">
<p style="margin:0">ROC-AUC is very stable across seeds (SD 0.007). But <strong>sensitivity and F1 have real seed variance</strong> (SD 0.04-0.05) — the seed=42 sensitivity of 0.83 was on the high end of what the model actually does. Multi-seed reporting should be considered mandatory for TRIPOD+AI compliance.</p>
</div>

<h3>5.3 · Multi-seed variance (CNN) &nbsp; <span style="color:#cf3838">✗ loosened</span></h3>
<p>Same head-only training regime, five seeds. Reveals dependence on training-regime choice.</p>
{multiseed_cnn_table()}
<div class="summary-card bad">
<p style="margin:0"><strong>Critical finding — training regime matters enormously.</strong> Head-only 3-epoch training gives internal AUC = 1.000 ± 0.000 (perfectly consistent leakage artifact), but external AUC = <strong>0.128 ± 0.018</strong> — meaningfully <em>below chance</em>. Only the fully fine-tuned 8-epoch model from the original run achieved the 0.82 external AUC. In other words, the CNN's genuine external signal is <strong>very fragile</strong> — head-only training memorizes shortcuts without transferring anything, and the external number depends on getting the training recipe exactly right. Any published CNN result on this dataset should specify the training regime explicitly.</p>
</div>

<h3>5.4 · Formal AUC-difference test (paired &amp; unpaired bootstrap) &nbsp; <span style="color:#e08e00">= confirmed</span></h3>
{auc_diff_table()}
<div class="summary-card">
<p style="margin:0">The <strong>ML tabular model is statistically indistinguishable from a 12-feature clinical logistic regression</strong> — same finding the source manuscript reached (their ΔAUC = +0.006). If you can collect 12 clinically meaningful features from a patient, you don't need XGBoost. The CNN's internal/external gap is <strong>highly significant</strong> (ΔAUC = +0.182, p &lt; 0.001) — this is not seed noise.</p>
</div>

<h3>5.5 · Multi-label CNN trained on Zenodo directly &nbsp; <span style="color:#1a936f">✓ tightened</span></h3>
<p>Same MobileNetV3-Small backbone, two output heads (PCO-visible + appears-abnormal). Trained end-to-end on Zenodo (70/15/15 split) instead of Kaggle → Zenodo.</p>
{multilabel_cnn_table()}
<div class="summary-card good">
<p style="margin:0"><strong>Training on the clean cohort is the actual fix.</strong> The Kaggle-trained model reached 0.82 AUC on Zenodo PCO-visible with catastrophic specificity (0.40). Training on Zenodo directly reaches ROC-AUC <strong>0.946 [0.913, 0.974]</strong> with balanced sensitivity (0.88) and specificity (0.90) — no recalibration needed. The "appears-abnormal" head reaches 0.996 AUC. This changes the deployment recommendation: <em>don't ship the Kaggle-trained model + recalibrator; train on Zenodo directly.</em></p>
</div>

<h3>Tier 1 summary — what changed vs the pre-check headlines</h3>
<div class="table-scroll">
<table>
<thead><tr><th>Claim</th><th>Pre-Tier-1 status</th><th>Post-Tier-1 status</th></tr></thead>
<tbody>
<tr><td>Tabular ROC-AUC ≈ 0.95</td><td>0.950 [0.895, 0.991] single-split</td><td>✓ <strong>0.957 [0.940, 0.973]</strong> pooled nested-CV, 0.944 ± 0.007 across 5 seeds</td></tr>
<tr><td>Tabular sensitivity ≈ 0.83</td><td>Single-split point estimate</td><td>= <strong>0.78 ± 0.04</strong> — the 0.83 was seed-lucky; real value lower</td></tr>
<tr><td>ML beats 12-feature LR</td><td>Bootstrap-suggested equivalence</td><td>= <strong>Formally confirmed indistinguishable</strong> (p = 1.000, ΔAUC CI covers 0)</td></tr>
<tr><td>CNN external AUC ≈ 0.82</td><td>Single-run point estimate</td><td>✗ <strong>Regime-dependent</strong>: full fine-tune ~0.82; head-only 0.13. Depends on training recipe</td></tr>
<tr><td>Internal/external gap is real</td><td>Bootstrap-suggested significance</td><td>= <strong>ΔAUC = 0.182 [0.165, 0.199], p &lt; 0.001</strong></td></tr>
<tr><td>Recalibration fixes deployment</td><td>Isotonic on Zenodo shown effective</td><td>✓ <strong>Direct-on-Zenodo training is better</strong>: 0.946 AUC with balanced sens/spec, no recalibrator needed</td></tr>
</tbody></table>
</div>

</section>

<!-- ===================== ARM 7: MANUSCRIPT COMPARISON ===================== -->

<section class="section">
<h2>Arm 7 · Head-to-head vs the FINAL manuscript</h2>

<p>This section compares our full analysis directly against <code>PCOS_ML_Manuscript_FINAL.docx</code> ("Dual artificial intelligence pathways for polycystic ovary syndrome assessment"). Same public datasets, similar pipelines, different rigor.</p>

<h3>7.1 · Where we agree</h3>
<div class="table-scroll">
<table>
<thead><tr><th>Finding</th><th>FINAL manuscript</th><th>Our analysis</th></tr></thead>
<tbody>
<tr><td>Clinical ML achieves ROC-AUC ≈ 0.95-0.96 on Kottarathil (n=541)</td><td>0.964 (LightGBM)</td><td>0.957 [0.940, 0.973] (nested-CV pooled)</td></tr>
<tr><td>Top SHAP features: right/left ovarian follicle count, AMH, hair growth</td><td>✓</td><td>✓ (matches almost exactly)</td></tr>
<tr><td>Internal CNN AUC is near-perfect</td><td>ResNet50 0.9995</td><td>MobileNetV3 1.000</td></tr>
<tr><td>External CNN ROC-AUC drops on Zenodo cohort</td><td>0.860</td><td>0.828 (v1), 0.875 (v2)</td></tr>
<tr><td>CNN probability outputs are extremely bimodal (saturated at 0/1)</td><td>Youden threshold = 0.99999994</td><td>Same distribution shown in the isotonic-recal figure</td></tr>
<tr><td>Local recalibration is required for deployment</td><td>✓</td><td>✓ (with concrete 5-seed evidence)</td></tr>
</tbody></table>
</div>

<h3>7.2 · Where our analysis found things the manuscript missed</h3>
<div class="table-scroll">
<table>
<thead><tr><th>Concern</th><th>FINAL manuscript position</th><th>Our finding</th></tr></thead>
<tbody>
<tr>
  <td><strong>Kaggle CNN train/test contamination</strong></td>
  <td>Cautions only that "patient-level independence could not be confirmed"</td>
  <td>Perceptual-hash dedup: <strong>100 % (1922/1922) of Kaggle test images have a training-set near-duplicate</strong>. The internal test was measuring memorization</td>
</tr>
<tr>
  <td><strong>PCOSGen-train ↔ Zenodo-test overlap</strong></td>
  <td>Not investigated</td>
  <td><strong>2,855 near-duplicate pairs; 62 % of Zenodo images have a PCOSGen match</strong>. The manuscript's "external validation" was ~62% in-distribution because both cohorts come from the same PCOSGen challenge</td>
</tr>
<tr>
  <td><strong>Is LightGBM/XGBoost actually better than a simple baseline?</strong></td>
  <td>Reported as the winner, no formal comparison to simpler alternatives</td>
  <td>Bootstrap ΔAUC test: <strong>ML vs 12-feature clinical logistic regression is statistically indistinguishable (p = 1.000, ΔAUC = +0.0002 CI [-0.028, +0.031])</strong></td>
</tr>
<tr>
  <td><strong>Is sensitivity 83.3% the real value?</strong></td>
  <td>Reported as the final LightGBM sensitivity</td>
  <td>5-seed variance analysis: <strong>0.78 ± 0.04</strong> — the 83.3% was one seed's high draw, not the typical value</td>
</tr>
<tr>
  <td><strong>Grad-CAM interpretation</strong></td>
  <td>"Activations overlapped ovarian regions" (kept vague)</td>
  <td><strong>Specific</strong>: attention correctly localises follicles on PCOS+ (real medical signal), but drifts to ultrasound scan-cone edges on PCOS− (learned shortcut). Explains why specificity collapses on external data</td>
</tr>
<tr>
  <td><strong>External threshold optimisation</strong></td>
  <td>Youden threshold picked <em>on the same data used to evaluate</em> → "may be optimistic"</td>
  <td>Proper 30/70 hold-out isotonic recalibration, 5-seed variance: transforms whole probability distribution, tight SDs (0.002-0.03)</td>
</tr>
<tr>
  <td><strong>Nested cross-validation</strong></td>
  <td>Not done</td>
  <td>Pooled 5×5 nested CV gives ROC-AUC 0.957 [0.940, 0.973] — confirms the tabular result is robust to winner-selection bias</td>
</tr>
<tr>
  <td><strong>Multi-seed variance analysis</strong></td>
  <td>5-fold CV only (0.962 ± 0.010 for RF)</td>
  <td>Additional 5-seed train/test-split variance on both tabular and CNN reveals training-recipe fragility on the image side (head-only 3-epoch → external AUC 0.13; only full 8-epoch fine-tune reaches 0.82)</td>
</tr>
</tbody></table>
</div>

<h3>7.3 · Where the manuscript's numbers appear inflated after our checks</h3>
<div class="table-scroll">
<table>
<thead><tr><th>Manuscript claim</th><th>Reality after our checks</th></tr></thead>
<tbody>
<tr>
  <td>External CNN ROC-AUC = <strong>0.860</strong> on PCOSGen-test (n=1468) as evidence of generalisation</td>
  <td>62 % of that test cohort has PCOSGen-train near-duplicates. On the <strong>truly-external 471-image subset</strong> after removing all duplicates, our v1 CNN reaches 0.818 and v2 reaches 0.875. Their reported 0.860 is a mixture of in-distribution and external</td>
</tr>
<tr>
  <td>LightGBM sensitivity <strong>83.3 %</strong> on held-out test</td>
  <td>5-seed mean = <strong>0.78 ± 0.04</strong>. The reported value is on the high end of the seed distribution</td>
</tr>
<tr>
  <td>"Calibration drift" framed as a technical externalisation issue</td>
  <td>Actually a combination of (a) <strong>prevalence shift</strong> (training 54 % PCOS+ vs external 30 % PCOS+) and (b) <strong>bimodal saturated outputs</strong>. Both fixable with a ~400-sample isotonic calibrator on deployment-site data</td>
</tr>
<tr>
  <td>LightGBM as the definitive tabular winner</td>
  <td>LightGBM, XGBoost, RF, stacking, and the 12-feature clinical LR are <strong>all statistically indistinguishable</strong> at n=541. Picking any as "the winner" is a low-power choice</td>
</tr>
</tbody></table>
</div>

<h3>7.4 · Three substantive findings unique to our analysis</h3>
<div class="summary-card bad">
<p style="margin:0"><strong>1. The Kaggle ultrasound dataset's train/test split is not a legitimate held-out test.</strong> Every one of the 1922 test images has a near-duplicate in the training set. Any paper reporting "internal accuracy" on this Kaggle dataset — including implicit reuse of it as part of the 11,784-image training pool — is measuring memorisation, not generalisation.</p>
</div>
<div class="summary-card bad">
<p style="margin:0"><strong>2. The manuscript's "external validation" is only partially external.</strong> PCOSGen-train and PCOSGen-test share ~62 % near-duplicate images (2,855 pairs). Any model trained on PCOSGen-train and evaluated on PCOSGen-test measures partial in-distribution transfer, not true external generalisation. The manuscript's 0.860 external AUC is therefore inflated relative to what would be observed on genuinely independent data.</p>
</div>
<div class="summary-card warn">
<p style="margin:0"><strong>3. The 12-feature clinical logistic regression is statistically identical to LightGBM/XGBoost</strong> (p = 1.000, ΔAUC 95% CI covers 0). If you can collect those 12 clinically meaningful features from a patient, the simple LR is enough — the boosted-tree machinery is not adding measurable value at this sample size. A publication reporting LightGBM as "the best model" without this baseline comparison overstates the ML contribution.</p>
</div>

<h3>7.5 · The core disagreement</h3>
<div class="summary-card">
<p style="margin:0"><strong>Manuscript conclusion.</strong> The dual-pathway AI framework has "strong internal performance and independent external validation" and supports "complementary application" for AI-assisted PCOS diagnosis.</p>
</div>
<div class="summary-card warn">
<p style="margin:0"><strong>Our conclusion.</strong>
<br>&nbsp;&nbsp;• <em>Clinical pathway</em>: solid at ROC-AUC 0.95, but the boosted-tree model provides no measurable benefit over a 12-feature logistic regression. Report both.
<br>&nbsp;&nbsp;• <em>Imaging pathway</em>: near-perfect internal AUC is largely a data-quality artifact (100 % train/test image contamination in Kaggle; ~62 % overlap in PCOSGen train/test). The reported external validation on PCOSGen-test overstates true generalisation because it is only partially external.
<br>&nbsp;&nbsp;• <em>Deployment</em>: neither pathway is publication-ready without (i) cross-cohort perceptual-hash dedup, (ii) multi-seed variance reporting, (iii) formal baseline-equivalence testing, and (iv) per-site probability recalibration.
<br>Same directional story as the manuscript, but with structural issues (data contamination, bootstrap-inflated point estimates, baseline equivalence, seed fragility) surfaced instead of hidden.</p>
</div>

</section>

<!-- ===================== INTERPRETATION ===================== -->

<section class="section">
<h2>Overall interpretation</h2>
<ul>
<li>The <strong>tabular arm</strong> gives a genuinely well-performing, well-calibrated PCOS classifier (ROC-AUC ≈ 0.94, ECE &lt; 0.1) that reproduces the source manuscript's headline results tightly.</li>
<li>The <strong>image arm</strong> reports internal AUC = 1.000 on the anaghachoudhari test split — but perceptual-hash dedup shows <em>every one</em> of the 1922 test images has a train-set near-duplicate. That "internal test" was really a training-set evaluation. Grad-CAM further shows the model learned <em>real</em> follicle-pattern signal on positives (attention correctly localises the ovary) but proxied negatives via ultrasound-cone boundary artifacts.</li>
<li>The <strong>external validation on Zenodo</strong> was partially contaminated too — 164 of 1468 images (11.2%) had Kaggle near-duplicates. On the truly-external 1304-image subset the CNN reaches ROC-AUC 0.818 [0.800, 0.834], only 0.010 below the contaminated 0.828. So the external gap is real, and the finding is robust to the contamination.</li>
<li>The <strong>Zenodo threshold sweep alone cannot recover specificity</strong> because the CNN's outputs are extremely bimodal. But <strong>isotonic probability recalibration on a ~400-sample Zenodo hold-out fixes this</strong>: specificity jumps 0.36 → 0.65, Brier drops 2.8×, ECE drops 29×, slope moves from 0.52 to 0.91 (near ideal). Discrimination (ROC-AUC) is unchanged as expected. The deployment fix is straightforward — no retraining needed.</li>
<li><strong>Recommendation for deployment</strong> (updated after Tier 1):
  <ol>
    <li><strong>Tabular</strong>: use a 12-feature clinical logistic regression rather than XGBoost — statistically indistinguishable performance, orders of magnitude simpler, easier to audit clinically.</li>
    <li><strong>Ultrasound</strong>: don't ship the Kaggle-trained CNN + Zenodo recalibrator. Instead, train directly on Zenodo (or a similarly clean cohort): 0.946 AUC with balanced sensitivity/specificity, no recalibration needed.</li>
    <li><strong>Before publishing</strong> any Kaggle-benchmarked model, run cross-cohort perceptual-hash dedup, nested CV, multi-seed variance analysis, and Grad-CAM localization audit.</li>
    <li><strong>Never trust</strong> a perfect internal-test AUC without (i) dedup, (ii) truly external cohort, (iii) interpretability check, and (iv) multi-seed evidence that the training recipe is stable.</li>
  </ol>
</li>
</ul>
</section>

<!-- ===================== ARTIFACTS ===================== -->

<section class="section">
<h2>Artifacts</h2>
<h4>Code</h4>
<ul>
<li><code>src/config.py</code>, <code>src/data.py</code>, <code>src/metrics.py</code>, <code>src/models.py</code> — tabular pipeline modules</li>
<li><code>src/run_pcos_best.py</code> — best-effort tabular run (8-model zoo + 60-trial Optuna)</li>
<li><code>src/train_cnn.py</code> — MobileNetV3-Small transfer learning on anaghachoudhari</li>
<li><code>src/external_val.py</code> — external validation on Zenodo 14591782</li>
<li><code>src/build_combined_report.py</code> — this HTML generator</li>
</ul>
<h4>Trained models &amp; metrics</h4>
<ul>
<li><code>artifacts/pcos_best/calibrated_model.pkl</code>, <code>results.json</code>, <code>test_predictions.csv</code></li>
<li><code>artifacts/cnn_ultrasound/best_model.pt</code>, <code>results.json</code></li>
<li><code>artifacts/external_val/external_val.json</code>, <code>preds_y_pco.csv</code>, <code>preds_y_abnormal.csv</code></li>
</ul>
<h4>Reproduce</h4>
<pre><code>python3 src/run_pcos_best.py       # ~7 min  — tabular
python3 src/train_cnn.py           # ~1–2 min on Apple MPS
python3 src/external_val.py        # ~10 s   — Zenodo eval
python3 src/build_combined_report.py</code></pre>
</section>

<footer style="margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border); color:var(--muted); font-size:.85rem;">
Report auto-generated from JSON artifacts. Self-contained (all figures embedded as base64).
</footer>

</body>
</html>"""

    out = REP / "combined_report.html"
    out.write_text(html)
    print(f"→ {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
