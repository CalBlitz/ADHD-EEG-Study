import os, re, math
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import pre_process as prep

# --------------------------- CONFIG ---------------------------------
OUT_DIR = getattr(prep, "OUT_DIR", "./outputs")
PSD_DIR = getattr(prep, "PSD_DIR", os.path.join(OUT_DIR, "psd80"))
ANALYSIS_DIR = getattr(prep, "ANALYSIS_DIR", os.path.join(OUT_DIR, "analysis"))
os.makedirs(ANALYSIS_DIR, exist_ok=True)


CONDS = list(getattr(prep, "CONDS", []))
if hasattr(prep, "TASKS"):
    TASKS = list(prep.TASKS.keys())
else:
    TASKS = []

# Preferred group order (blink-resistant first)
PREFERRED_GROUPS = list(getattr(prep, "PREFERRED_GROUPS",
                                ["lateral_frontal", "all", "frontopolar", "ungrouped", None]))

# Rest vs non-rest
REST_TASKS = list(getattr(prep, "REST_TASKS", ["Rest"]))

# ---------- Bands: original “initials” ----------
def _bands_from_prep():
    B = getattr(prep, "BANDS", None)
    if isinstance(B, dict):
        return B

    out = {}
    for name in ["delta","theta","alpha","beta","lgam","low_gamma"]:
        if hasattr(B, name):
            out["low_gamma" if name in ("lgam","low_gamma") else name] = getattr(B, name)

    # Ensure all expected keys exist
    out.setdefault("delta",(0.5,4.0))
    out.setdefault("theta",(4.0,8.0))
    out.setdefault("alpha",(8.0,12.0))
    out.setdefault("beta",(12.0,30.0))
    out.setdefault("low_gamma",(30.0,40.0))
    return out

BANDS = _bands_from_prep()

# ---------- Helpers ----------
def _scan_psd_inventory():
    """
    Parses PSD filenames to discover conditions, tasks, and groups.
    Supports:
      {cond}_{task}.csv
      {cond}_{task}__{group}.csv
    """
    conds, tasks, groups = set(), set(), set()
    if not os.path.isdir(PSD_DIR):
        return [], [], []
    for fn in os.listdir(PSD_DIR):
        if not fn.endswith(".csv"):
            continue
        name = fn[:-4]
        group = None
        left = name
        if "__" in name:
            left, group = name.split("__", 1)
        if "_" not in left:
            continue
        cond, task = left.split("_", 1)
        conds.add(cond); tasks.add(task); groups.add(group if group is not None else "ungrouped")
    # Canonicalize ordering
    conds = sorted(conds, key=lambda c: {"baseline":0,"medication":1,"meditation":2}.get(c, 99))
    tasks = sorted(tasks, key=lambda t: ["Rest","Math","Music","Video","Retrieval","ColorCounting"].index(t)
                   if t in ["Rest","Math","Music","Video","Retrieval","ColorCounting"] else 99)
    # Preferred groups come first
    groups = [g for g in PREFERRED_GROUPS if g in groups] + [g for g in groups if g not in PREFERRED_GROUPS]
    return conds, tasks, groups

def _ensure_inventory():
    global CONDS, TASKS
    if not (CONDS and TASKS):
        conds, tasks, _ = _scan_psd_inventory()
        if not CONDS: CONDS = conds
        if not TASKS: TASKS = tasks
    return CONDS, TASKS

def _try_psd_path(cond, task, group):
    if group is None or group == "ungrouped":
        p = os.path.join(PSD_DIR, f"{cond}_{task}.csv")
        return p if os.path.exists(p) else None
    p = os.path.join(PSD_DIR, f"{cond}_{task}__{group}.csv")
    return p if os.path.exists(p) else None

# ---------- IO + math helpers ----------
def _read_psd_any(cond, task):
    """Yield (group, freqs, matrix) for every available group file for this (cond, task)."""
    freqs = None
    yielded = False
    # Try all present groups for this pair
    for g in PREFERRED_GROUPS:
        p = _try_psd_path(cond, task, g)
        if p:
            df = pd.read_csv(p)
            if freqs is None:
                freqs = np.array([float(c.replace("Hz","")) for c in df.columns])
            yield (g if g is not None else "ungrouped"), freqs, df.values
            yielded = True
    # Also scan any other groups not listed in preference (if they exist)
    base = f"{cond}_{task}__"
    if os.path.isdir(PSD_DIR):
        for fn in os.listdir(PSD_DIR):
            if fn.startswith(base) and fn.endswith(".csv"):
                g = fn[len(base):-4]
                if g in PREFERRED_GROUPS:
                    continue  # already handled
                p = os.path.join(PSD_DIR, fn)
                df = pd.read_csv(p)
                if freqs is None:
                    freqs = np.array([float(c.replace("Hz","")) for c in df.columns])
                yield g, freqs, df.values
                yielded = True
    # If only ungrouped exists and wasn't caught above
    p_ung = _try_psd_path(cond, task, "ungrouped")
    if (not yielded) and p_ung:
        df = pd.read_csv(p_ung)
        freqs = np.array([float(c.replace("Hz","")) for c in df.columns])
        yield "ungrouped", freqs, df.values

def _integrate_band(freqs, P_rows, lo, hi):
    """Integrate power in [lo, hi) for each row using Riemann sum (equal bin widths)."""
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.full(P_rows.shape[0], np.nan)
    df = np.mean(np.diff(freqs[mask])) if np.sum(mask) > 1 else (hi - lo)
    return np.sum(P_rows[:, mask], axis=1) * df

def _paired_stats(a, b, label_a="A", label_b="B"):
    """Paired t, Wilcoxon, Cohen's dz. Returns a dict with means and deltas."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out = dict(n=len(a), label_a=label_a, label_b=label_b)
    if len(a) >= 2:
        t, p = ttest_rel(a, b)
        try:
            w, pw = wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            w, pw = np.nan, np.nan
        diff = b - a
        dz = np.mean(diff) / (np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan)
        out.update(t=float(t), p=float(p), wilcoxon_W=float(w), wilcoxon_p=float(pw), cohens_dz=float(dz))
    else:
        out.update(t=np.nan, p=np.nan, wilcoxon_W=np.nan, wilcoxon_p=np.nan, cohens_dz=np.nan)
    out["mean_a"] = float(np.mean(a)) if len(a) else np.nan
    out["mean_b"] = float(np.mean(b)) if len(b) else np.nan
    out["mean_diff_b_minus_a"] = out["mean_b"] - out["mean_a"]
    out["pct_diff_b_minus_a"] = (100.0 * out["mean_diff_b_minus_a"] / out["mean_a"]) if (out["mean_a"] and out["mean_a"]>0) else np.nan
    return out

def _pick_group_available(df, candidates):
    """Return the first group in `candidates` contained in df['group']."""
    avail = set(df["group"].unique())
    for g in candidates:
        if g in avail:
            return g
    return next(iter(avail)) if avail else None

# Band table by group
def _assemble_band_table():
    """
    Returns a DataFrame with rows: (condition, task, day, group, delta, theta, alpha, beta, low_gamma, tbr, total_power_0p5_40)
    Saves to ./outputs/analysis/band_table_by_group.csv
    """
    conds, tasks = _ensure_inventory()
    rows = []
    for cond in conds:
        for task in tasks:
            any_yield = False
            for group, freqs, M in _read_psd_any(cond, task):
                any_yield = True
                # compute bands per day
                delta = _integrate_band(freqs, M, *BANDS["delta"])
                theta = _integrate_band(freqs, M, *BANDS["theta"])
                alpha = _integrate_band(freqs, M, *BANDS["alpha"])
                beta  = _integrate_band(freqs, M, *BANDS["beta"])
                lgam  = _integrate_band(freqs, M, *BANDS["low_gamma"])
                total = _integrate_band(freqs, M, 0.5, 40.0)
                tbr   = theta / beta
                for d in range(M.shape[0]):
                    rows.append(dict(
                        condition=cond, task=task, day=d+1, group=group,
                        delta=delta[d], theta=theta[d], alpha=alpha[d], beta=beta[d],
                        low_gamma=lgam[d], tbr=tbr[d], total_power_0p5_40=total[d]
                    ))
            if not any_yield:
                # nothing available for this (cond, task)
                pass
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ANALYSIS_DIR, "band_table_by_group.csv"), index=False)
    return df


def plot_bar_with_error(ax, means, sems, labels, title, ylabel):
    x = np.arange(len(means))
    ax.bar(x, means, yerr=sems, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title); ax.set_ylabel(ylabel)

# ----------------------------- Hypothesis testing- Main function----------------------------
def run_hypothesis_tests(df):
    """
    H1: Baseline -> Medication  (expect theta↓, beta↑, TBR↓)
        Use a *blink-resistant* group if available (lateral_frontal preferred).
    H2: Baseline -> Meditation  (expect theta↑, alpha↑)
        Prefer frontopolar for theta, lateral_frontal for alpha; fall back gracefully.
    Saves:
      - H1_results_by_group.csv  (+ bar plots per taskset)
      - H2_results_by_group.csv  (+ bar plots per taskset)
    """
    _, tasks = _ensure_inventory()
    nonrest_default = [t for t in tasks if t not in REST_TASKS]
    NONREST_TASKS = list(getattr(prep, "NONREST_TASKS", nonrest_default))
    
    # Discover available groups in df
    groups_avail = list(df["group"].dropna().unique())
    if not groups_avail:
        raise RuntimeError("No groups found in band table. Did pre_process write PSDs?")

    # Preferred order
    preferred = list(getattr(prep, "PREFERRED_GROUPS", PREFERRED_GROUPS))

    # Group choices
    group_h1 = _pick_group_available(df, preferred)
    theta_pref = ["frontopolar"] + preferred
    alpha_pref = ["lateral_frontal"] + preferred
    group_h2_theta = _pick_group_available(df, theta_pref) or group_h1
    group_h2_alpha = _pick_group_available(df, alpha_pref) or group_h1

    print(f"[H1] Using group: {group_h1}")
    print(f"[H2] Theta group: {group_h2_theta} | Alpha group: {group_h2_alpha}")

    def _agg_by(cond, sub):
        tmp = sub[sub["condition"]==cond].groupby("day")[["theta","alpha","beta","tbr"]].mean().sort_index()
        return tmp

    # ========== Hypothesis 1 ==========
    # Baseline → Medication: expect theta↓, beta↑, TBR↓
    h1_rows = []
    subset_h1 = df[df["group"] == group_h1].copy()
    tasksets_h1 = [
        ("Rest only", REST_TASKS),
        ("All non-rest", NONREST_TASKS),
        ("All tasks", REST_TASKS + NONREST_TASKS),
    ]
    for tname, tasklist in tasksets_h1:
        sub = subset_h1[subset_h1["task"].isin(tasklist)]
        base = _agg_by("baseline", sub)
        med  = _agg_by("medication", sub)
        join = base.join(med, lsuffix="_base", rsuffix="_med", how="inner")
        if join.empty:
            continue
        for metric in ["theta","beta","tbr"]:
            stats = _paired_stats(join[f"{metric}_base"], join[f"{metric}_med"], "Baseline", "Medication")
            stats.update(metric=metric, taskset=tname, group_used=group_h1)
            h1_rows.append(stats)

        # Bar plots (mean±SEM)
        fig, ax = plt.subplots(1,3, figsize=(9,3))
        for i, metric in enumerate(["theta","beta","tbr"]):
            means = [join[f"{metric}_base"].mean(), join[f"{metric}_med"].mean()]
            sems  = [join[f"{metric}_base"].sem(),  join[f"{metric}_med"].sem()]
            x = np.arange(2); width = 0.6
            ax[i].bar(x, means, yerr=sems, capsize=5)
            ax[i].set_xticks(x); ax[i].set_xticklabels(["Baseline","Medication"])
            ax[i].set_title(f"H1 – {tname}\n{metric.upper()} (grp: {group_h1})")
            ax[i].set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"H1_{tname.replace(' ','_')}_group-{group_h1}.png"), dpi=160)
        plt.close(fig)

    pd.DataFrame(h1_rows).to_csv(os.path.join(ANALYSIS_DIR, "H1_results_by_group.csv"), index=False)

    # ========== Hypothesis 2 ==========
    # Baseline → Meditation: expect theta↑ and alpha↑
    h2_rows = []
    tasksets_h2 = [
        ("Rest only", REST_TASKS),
        ("All non-rest", NONREST_TASKS),
        ("All tasks", REST_TASKS + NONREST_TASKS),
    ]
    for tname, tasklist in tasksets_h2:
        # Theta (frontopolar-pref)
        sub_t = df[(df["group"] == group_h2_theta) & (df["task"].isin(tasklist))]
        base_t = _agg_by("baseline", sub_t)
        medt_t = _agg_by("meditation", sub_t)
        join_t = base_t.join(medt_t, lsuffix="_base", rsuffix="_medit", how="inner")
        if not join_t.empty:
            st = _paired_stats(join_t["theta_base"], join_t["theta_medit"], "Baseline", "Meditation")
            st.update(metric="theta", taskset=tname, group_used=group_h2_theta)
            h2_rows.append(st)

        # Alpha (lateral-frontal-pref)
        sub_a = df[(df["group"] == group_h2_alpha) & (df["task"].isin(tasklist))]
        base_a = _agg_by("baseline", sub_a)
        medt_a = _agg_by("meditation", sub_a)
        join_a = base_a.join(medt_a, lsuffix="_base", rsuffix="_medit", how="inner")
        if not join_a.empty:
            sa = _paired_stats(join_a["alpha_base"], join_a["alpha_medit"], "Baseline", "Meditation")
            sa.update(metric="alpha", taskset=tname, group_used=group_h2_alpha)
            h2_rows.append(sa)

        # Plots: THETA (left) & ALPHA (right)
        fig, ax = plt.subplots(1,2, figsize=(7.2,3))
        if not join_t.empty:
            means = [join_t["theta_base"].mean(), join_t["theta_medit"].mean()]
            sems  = [join_t["theta_base"].sem(),  join_t["theta_medit"].sem()]
            x = np.arange(2); ax[0].bar(x, means, yerr=sems, capsize=5)
            ax[0].set_xticks(x); ax[0].set_xticklabels(["Baseline","Meditation"])
            ax[0].set_title(f"H2 – {tname}\nTHETA (grp: {group_h2_theta})"); ax[0].set_ylabel("theta")
        if not join_a.empty:
            means = [join_a["alpha_base"].mean(), join_a["alpha_medit"].mean()]
            sems  = [join_a["alpha_base"].sem(),  join_a["alpha_medit"].sem()]
            x = np.arange(2); ax[1].bar(x, means, yerr=sems, capsize=5)
            ax[1].set_xticks(x); ax[1].set_xticklabels(["Baseline","Meditation"])
            ax[1].set_title(f"H2 – {tname}\nALPHA (grp: {group_h2_alpha})"); ax[1].set_ylabel("alpha")
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"H2_{tname.replace(' ','_')}_theta-{group_h2_theta}_alpha-{group_h2_alpha}.png"), dpi=160)
        plt.close(fig)

    pd.DataFrame(h2_rows).to_csv(os.path.join(ANALYSIS_DIR, "H2_results_by_group.csv"), index=False)

def main():
    # 1) Build band table across ALL available groups from PSD files
    df = _assemble_band_table()

    # 2) Run group-aware H1/H2 hypothesis tests + plots
    run_hypothesis_tests(df)

    print(f"Hypothesis analysis completed.\n→ {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()