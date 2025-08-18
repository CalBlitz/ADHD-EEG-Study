import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pre_process as prep

# ---------- Resolve paths and config from pre_process (with safe fallbacks) ----------
OUT_DIR = getattr(prep, "OUT_DIR", "./outputs")
PSD_DIR = getattr(prep, "PSD_DIR", os.path.join(OUT_DIR, "psd80"))
ANALYSIS_DIR = getattr(prep, "ANALYSIS_DIR", os.path.join(OUT_DIR, "analysis"))
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Ensure we have the expected tasks from pre_process
if hasattr(prep, "TASKS"):
    TASKS = list(prep.TASKS.keys())
else:
    TASKS = []

CONDS = list(getattr(prep, "CONDS", []))
PREFERRED_GROUPS = list(getattr(prep, "PREFERRED_GROUPS", ["lateral_frontal", "all", "frontopolar", None]))
REST_TASKS = list(getattr(prep, "REST_TASKS", ["Rest"]))

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
    Parse PSD filenames to discover conditions, tasks, and groups.
    Supported patterns:
      {cond}_{task}.csv
      {cond}_{task}__{group}.csv
    """
    conds, tasks, groups = set(), set(), set()
    if not os.path.isdir(PSD_DIR):
        return [], [], []
    for fn in os.listdir(PSD_DIR):
        if not fn.endswith(".csv"):
            continue
        name = fn[:-4]  # strip .csv
        # Split on group suffix if present
        if "__" in name:
            left, group = name.split("__", 1)
            groups.add(group)
        else:
            left, group = name, None
            groups.add(None)
        # Split left into cond and task (1 underscore split)
        # Assumes cond and task contain no extra underscores (true for our set)
        parts = left.split("_", 1)
        if len(parts) != 2:
            continue
        cond, task = parts
        conds.add(cond)
        tasks.add(task)
    # Canonical order for conditions if present
    conds = sorted(conds, key=lambda c: {"baseline":0,"medication":1,"meditation":2}.get(c, 99))
    tasks = sorted(tasks, key=lambda t: ["Rest","Math","Music","Video","Retrieval","ColorCounting"].index(t) if t in ["Rest","Math","Music","Video","Retrieval","ColorCounting"] else 99)
    groups = [g for g in PREFERRED_GROUPS if g in groups] + [g for g in groups if g not in PREFERRED_GROUPS]
    return conds, tasks, groups

def _ensure_inventory():
    global CONDS, TASKS
    if not CONDS or not TASKS:
        conds, tasks, groups = _scan_psd_inventory()
        if not CONDS:
            CONDS = conds
        if not TASKS:
            TASKS = tasks
    return CONDS, TASKS

def _try_psd_path(cond, task, group):
    if group is None:
        p = os.path.join(PSD_DIR, f"{cond}_{task}.csv")
        return p if os.path.exists(p) else None
    p = os.path.join(PSD_DIR, f"{cond}_{task}__{group}.csv")
    return p if os.path.exists(p) else None

def _pick_existing_psd(cond, task):
    for g in PREFERRED_GROUPS:
        p = _try_psd_path(cond, task, g)
        if p:
            return p, g
    return None, None

def _parse_freq_cols(cols):
    # allow "0.5", "0.5Hz", "f0p5", etc.
    freqs = []
    for c in cols:
        s = str(c).replace("Hz","").replace("f","").replace("F","").replace("p",".")
        m = re.findall(r"[\d\.]+", s)
        freqs.append(float(m[0]) if m else np.nan)
    return np.array(freqs)

def _load_psd_csv(path):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    freqs = _parse_freq_cols(df.columns)
    P = df.values.astype(float)    # rows = days, cols = freqs
    return freqs, P

def read_psd(cond, task):
    """
    Returns: freqs (80,), matrix (n_days, 80), group_used
    """
    path, g = _pick_existing_psd(cond, task)
    if path is None:
        return None, None, None
    df = pd.read_csv(path)
    freqs = np.array([float(c.replace("Hz","")) for c in df.columns])
    return freqs, df.values, (g if g is not None else "ungrouped")

def integrate_band(freqs, P_rows, lo, hi):
    """Integrate power in [lo, hi) per row; bins are ~0.5 Hz."""
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.full(P_rows.shape[0], np.nan)
    df = np.mean(np.diff(freqs[mask])) if np.sum(mask) > 1 else (hi - lo)
    return np.sum(P_rows[:, mask], axis=1) * df

def pearson_corr_vec(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: 
        return np.nan
    return pearsonr(a[m], b[m])[0]

def _annotate_bands(ax):
    bands = [("Delta",(0.5,4)),("Theta",(4,8)),("Alpha",(8,12)),("Beta",(12,30)),("Gamma",(30,40))]
    edges = sorted({b[1][0] for b in bands} | {b[1][1] for b in bands})
    for name,(lo,hi) in bands:
        ax.axvspan(lo, hi, alpha=0.08, zorder=0)
        ax.text((lo+hi)/2, ax.get_ylim()[1]/1.6, name, ha="center", va="top", fontsize=8)
    for x in edges:
        ax.axvline(x, ls="--", lw=0.8, alpha=0.6, color="k")

# ---------- Main analyses ----------
def compute_band_table():
    """
    Long-form table: (condition, task, day, group, delta, theta, alpha, beta, low_gamma, tbr, total_0p5_40)
    """
    rows = []
    conds, tasks = _ensure_inventory()
    for cond in conds:
        for task in tasks:
            freqs, M, group = read_psd(cond, task)
            if M is None:
                continue
            bp = {name: integrate_band(freqs, M, *rng) for name, rng in BANDS.items()}
            total = integrate_band(freqs, M, 0.5, 40.0)
            tbr = bp["theta"] / bp["beta"]
            for d in range(M.shape[0]):
                rows.append({
                    "condition": cond, "task": task, "day": d+1, "group": group,
                    "delta": bp["delta"][d], "theta": bp["theta"][d], "alpha": bp["alpha"][d],
                    "beta": bp["beta"][d], "low_gamma": bp["low_gamma"][d],
                    "tbr": tbr[d], "total_power_0p5_40": total[d],
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ANALYSIS_DIR, "band_table.csv"), index=False)
    return df

def inter_session_correlation(save_png=True):
    """
    Day×Day Pearson correlation matrices between PSD vectors for each (cond, task).
    """
    conds, tasks = _ensure_inventory()
    results = {}
    for cond in conds:
        for task in tasks:
            path, group = _pick_existing_psd(cond, task)
            if not path:
                continue
            M = pd.read_csv(path).values
            n = M.shape[0]
            C = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    C[i, j] = pearson_corr_vec(M[i], M[j])
            results[(cond, task)] = C

            if save_png:
                plt.figure(figsize=(4,3))
                plt.imshow(C, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                plt.colorbar(label="Correlation")
                plt.title(f"{cond.capitalize()} – {task}\nInter-session similarity")
                plt.xlabel("Day"); plt.ylabel("Day")
                ticks = range(n)
                plt.xticks(ticks, [t+1 for t in ticks]); plt.yticks(ticks, [t+1 for t in ticks])
                plt.tight_layout()
                plt.savefig(os.path.join(ANALYSIS_DIR, f"{cond}_{task}_corr.png"), dpi=160)
                plt.close()
    return results

def similarity_to_day1_trends():
    """
    For each task, plot per-condition correlation to Day 1 PSD over days.
    """
    conds, tasks = _ensure_inventory()
    for task in tasks:
        plt.figure(figsize=(6,4))
        for cond in conds:
            path, group = _pick_existing_psd(cond, task)
            if not path: 
                continue
            M = pd.read_csv(path).values
            r = [pearson_corr_vec(M[0], M[i]) for i in range(M.shape[0])]
            plt.plot(range(1, len(r)+1), r, marker="o", label=cond.capitalize())
        plt.title(f"{task} – Similarity to Day 1 (PSD)")
        plt.xlabel("Day"); plt.ylabel("Correlation")
        plt.ylim(0,1); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"{task}_to_day1_trends.png"), dpi=160)
        plt.close()

def per_task_band_summary(df):
    """
    Wide summary: mean/std/sem/median of bands by condition×task.
    """
    g = df.groupby(["condition","task"])[["delta","theta","alpha","beta","low_gamma","tbr"]]
    summary = g.agg(["mean","std","sem","median"])
    summary.to_csv(os.path.join(ANALYSIS_DIR, "per_task_band_summary.csv"))
    return summary

# ---------- Rest vs Non-Rest visualizations ----------
def save_psd_per_task_over_conditions(group="lateral_frontal"):
    saved = []
    for task in TASKS:
        curves = []
        freqs_ref = None
        for cond in CONDS:
            path = os.path.join(PSD_DIR, f"{cond}_{task}__{group}.csv")
            if not os.path.exists(path):
                print(f"[skip] missing PSD: {path}")
                continue

            df = pd.read_csv(path)

            # --- keep only numeric frequency columns (drop 'Unnamed: 0', etc.)
            cols, freqs = [], []
            for c in df.columns:
                m = re.findall(r"[-+]?\d*\.?\d+", str(c).replace("Hz", ""))
                if not m:
                    continue
                try:
                    freqs.append(float(m[0]))
                    cols.append(c)
                except ValueError:
                    pass
            if not freqs:
                print(f"[skip] no numeric freq cols in {path}")
                continue

            f = np.array(freqs, dtype=float)
            P = df[cols].values.astype(float)   # rows = days, cols = freqs

            # set reference freqs from the first successfully loaded file
            if freqs_ref is None:
                freqs_ref = f

            # aggregate across days
            y = np.nanmedian(P, axis=0)  # or np.nanmean(P, axis=0)
            curves.append((cond.capitalize(), y))

        if not curves or freqs_ref is None:
            print(f"[warn] no curves for task {task}")
            continue

        plt.figure(figsize=(8, 4))
        for lbl, y in curves:
            plt.semilogy(freqs_ref, y, label=lbl)
        _annotate_bands(plt.gca())  # your existing helper
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (µV²/Hz)")
        plt.title(f"{task} — {group.replace('_',' ').title()}  (median across days)")
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        fn = os.path.join(ANALYSIS_DIR, f"psd_{task}__{group}__median.png")
        plt.savefig(fn, dpi=150)
        plt.show()
        saved.append(fn)

    print("Saved PSD overlays:", *saved, sep="\n- ")
    return saved


def rest_vs_nonrest_analysis(df):
    """
    Creates:
      - rest_means.csv, nonrest_means.csv, rest_vs_nonrest_diff.csv
      - rest_vs_nonrest_{metric}.png (paired bars REST vs NONREST per condition)
      - rest_vs_nonrest_differences.png (grid of NONREST-REST deltas)
    """
    # Derive NONREST from discovered tasks
    _, tasks = _ensure_inventory()
    nonrest_default = [t for t in tasks if t not in REST_TASKS]
    NONREST_TASKS = list(getattr(prep, "NONREST_TASKS", nonrest_default))

    # Choose a single group (prefer order in PREFERRED_GROUPS)
    groups_available = list(df["group"].unique())
    chosen_group = next((g for g in PREFERRED_GROUPS if g in groups_available), groups_available[0])
    sub = df[df["group"] == chosen_group].copy()

    # Aggregate per condition & day
    metrics = ["delta","theta","alpha","beta","low_gamma","tbr"]
    rest_day = sub[sub["task"].isin(REST_TASKS)].groupby(["condition","day"])[metrics].mean()
    nonr_day = sub[sub["task"].isin(NONREST_TASKS)].groupby(["condition","day"])[metrics].mean()

    # Means across days (for CSVs)
    rest_means = rest_day.groupby("condition").mean()
    nonr_means = nonr_day.groupby("condition").mean()
    rest_means.add_prefix("rest_").to_csv(os.path.join(ANALYSIS_DIR, "rest_means.csv"))
    nonr_means.add_prefix("nonrest_").to_csv(os.path.join(ANALYSIS_DIR, "nonrest_means.csv"))
    (nonr_means - rest_means).to_csv(os.path.join(ANALYSIS_DIR, "rest_vs_nonrest_diff.csv"))

    # Paired bar plots (REST vs NONREST) with SEM across days
    conditions = [c for c in ["baseline","medication","meditation"] if c in rest_means.index]
    width = 0.35
    for m in metrics:
        fig, ax = plt.subplots(figsize=(6.8,4))
        x = np.arange(len(conditions))
        # compute means & sem across days per condition
        r_mean = rest_day[m].groupby("condition").mean().reindex(conditions)
        n_mean = nonr_day[m].groupby("condition").mean().reindex(conditions)
        r_sem  = rest_day[m].groupby("condition").sem().reindex(conditions)
        n_sem  = nonr_day[m].groupby("condition").sem().reindex(conditions)

        ax.bar(x - width/2, r_mean.values, width, yerr=r_sem.values, capsize=5, label="Rest")
        ax.bar(x + width/2, n_mean.values, width, yerr=n_sem.values, capsize=5, label="Non-rest")
        ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in conditions])
        ax.set_ylabel(m)
        ax.set_title(f"Rest vs Non-rest – {m}  (group: {chosen_group})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"rest_vs_nonrest_{m}.png"), dpi=160)
        plt.close()

    # Grid of NONREST - REST differences
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.ravel()
    diff = (nonr_means - rest_means).reindex(conditions)
    for ax, m in zip(axes, metrics):
        ax.bar([c.capitalize() for c in conditions], diff[m].values)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"Non-rest minus Rest – {m}")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "rest_vs_nonrest_differences.png"), dpi=160)
    plt.close()

# ---------- Main ----------
def main():
    # 1) Build band table
    df = compute_band_table()

    # 2) Per-task summaries
    per_task_band_summary(df)

    # 3) Day-to-day PSD similarity analyses
    inter_session_correlation(save_png=True)
    similarity_to_day1_trends()

    # 4) Rest vs Non-Rest visualizations
    save_psd_per_task_over_conditions()
    rest_vs_nonrest_analysis(df)

    print(f"Analysis complete.\nFiles savd at {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()