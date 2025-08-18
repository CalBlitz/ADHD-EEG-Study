import os, re, math, glob, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, iirnotch, welch, detrend
from scipy.signal import butter, filtfilt, iirnotch, welch, detrend
from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# --------------------------- CONFIG ---------------------------------
DATA_DIR = "./data"
OUT_DIR  = "./output"

FS = 200  # Hz

# task windows (start sec, length sec).
TASKS = {
    "Rest":           (27, 30),
    "Math":           (61, 30),
    "Music":          (95, 30),
    "Video":          (132, 30),
    "Retrieval":      (182, 30),
    "ColorCounting":  (238, 30),  # first 30s of Task 6
}
REST_TASKS = ["Rest", "Music", "Retrieval"]
NONREST_TASKS = ["Math", "Video", "ColorCounting"]

# Channel Specifications
CHANNEL_LABELS = ["Fp1","Fp2","F7","F8"]
CHANNEL_GROUPS = {
    "all": [0,1,2,3],
    "frontopolar": [0,1],        # Fp1, Fp2 (blink-prone, best for frontal theta)
    "lateral_frontal": [2,3],    # F7, F8  (less blink; use for TBR/beta/alpha)
}
PREFERRED_GROUPS = ["lateral_frontal", "all", "frontopolar", None]
UNITS = "uV"  # set to "counts" if the recordings are in integer counts
SCALE_UV_PER_COUNT = 0.001869917138805  #Scaling Factor for converrting counts → microvolts

# Clean-epoch search inside each 30s task
CLEAN_EPOCH_SEC = 15
SPIKE_THRESHOLD_UV = 50.0  # ±50 µV

# Re-reference and combine the 4 channels
REREF = "car"
# PCA tracks the strongest shared component while down-weighting noisy sensors
COMBINE_METHOD = "pca1" # "mean", "rms", "pca1", "ch1", "ch2", "ch3", "ch4"

# Notch/band limits
NOTCH_HZ = 50.0
BANDPASS = (1, 40.0)

# Welch PSD settings (Hann, 2s, 50% overlap, median average)
N_PERSEG = int(2.0 * FS)     # 400 samples (2s * 200Hz)
N_OVERLP = int(0.5 * N_PERSEG)  # 200 samples (50% overlap)

# --------------------------------------------------------------------

warnings.simplefilter("ignore", RuntimeWarning)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- Find "type" and "day" from file name ----------
def parse_type_and_day(fname: str):
    base = os.path.basename(fname).lower()
    # map a variety of spellings to canonical names
    if   any(x in base for x in ["baseline", "base"]): cond = "baseline"
    elif any(x in base for x in ["medication", "medicine", "med"]): cond = "medication"
    elif any(x in base for x in ["meditaion", "om_", "meditate"]): cond = "meditation"
    else: cond = "unknown"

    m = re.search(r"day[_\- ]?(\d+)", base)
    day = int(m.group(1)) if m else None
    return cond, day

# ------------------- Load packets --------------------
def load_ganglion_csv(path):
    """ Load CSV from Ganglion device.
    Returns ndarray shape (N, 4) for 4 EEG channels in µV.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 5:
        raise ValueError(f"{path}: expected at least 5 columns (packet + 4 channels).")
    df.columns = ['packet','ch1','ch2','ch3','ch4'] + [f'extra{i}' for i in range(df.shape[1]-5)]
    x = df[['ch1','ch2','ch3','ch4']].astype(float).to_numpy()

    if UNITS.lower() == "counts":
        x = x * float(SCALE_UV_PER_COUNT)   # 0.001869917138805 µV per count
    return x  # shape (N,4) at full 200 Hz

# ------------------------- Preprocessing ----------------------------
def apply_notch_and_bandpass(x, fs=FS, notch=NOTCH_HZ, band=BANDPASS):

    # notch
    b_notch, a_notch = iirnotch(w0=notch/(fs/2), Q=30)
    x = filtfilt(b_notch, a_notch, x, axis=0)

    # band-pass
    low, high = band
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    x = filtfilt(b, a, x, axis=0)

    # detrend
    x = detrend(x, axis=0, type='constant')
    return x

def fit_ICA(raw,
            fs=FS,
            band=(1.0, 40.0),
            n_components=4,
            max_iter=5000,
            tol=1e-5,
            random_state=42):
    """
    (1) Filter + standardize, (2) fit ICA and compute EOG-based scores.

    Returns:
        fit (dict): {
            'raw_f': filtered data (N,4),
            'X': same as raw_f (np.ndarray),
            'Xz': z-scored channels,
            'ica': fitted FastICA object,
            'S': sources (N, n_components),
            'A': mixing matrix (4, n_components),
            'eog_proxy': normalized Fp mean,
            'rows': list of [comp, corr_eog, lf_ratio, kurtosis, fp_ratio],
            'scores': pandas DataFrame of the rows,
            'fs': fs
        }
    """
    # 1) Fit ICA
    raw_f = apply_notch_and_bandpass(raw, fs=fs, band=band)
    X  = np.asarray(raw_f)
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    ica = FastICA(n_components=n_components, algorithm="parallel",
                  whiten="unit-variance", fun="logcosh",
                  max_iter=max_iter, tol=tol, random_state=random_state)
    S = ica.fit_transform(Xz)   # (N, n_comp)
    A = ica.mixing_             # (4, n_comp)

    # 2) EOG proxy and helper scores
    eog_proxy = (Xz[:, 0] + Xz[:, 1]) / 2.0
    eog_proxy = (eog_proxy - eog_proxy.mean()) / (eog_proxy.std() + 1e-9)

    def lf_ratio(sig, fs=fs, split=3.0):
        f, P = welch(sig, fs=fs, window="hann",
                     nperseg=int(2*fs), noverlap=int(fs),
                     average="median", detrend=False, scaling="density")
        m1 = (f >= 0.5) & (f < split)
        m2 = (f >= split) & (f <= 40.0)
        return (np.trapz(P[m1], f[m1]) + 1e-12) / (np.trapz(P[m2], f[m2]) + 1e-12)

    rows = []
    for i in range(S.shape[1]):
        si   = (S[:, i] - S[:, i].mean()) / (S[:, i].std() + 1e-9)
        corr = np.corrcoef(si, eog_proxy)[0, 1]
        lfr  = lf_ratio(si, fs, split=3.0)
        k    = kurtosis(si, fisher=True, bias=False)

        # frontal loading (Fp vs F7/F8) from mixing matrix column i
        fp_load = np.sum(np.abs(A[[0, 1], i]))  # Fp1,Fp2
        lf_load = np.sum(np.abs(A[[2, 3], i]))  # F7,F8
        fp_ratio = fp_load / (lf_load + 1e-9)

        rows.append([i, corr, lfr, k, fp_ratio])

    scores = pd.DataFrame(rows, columns=["comp", "corr_eog", "lf_ratio<3Hz", "kurtosis", "fp_ratio(Fp/F7-8)"])

    return dict(raw_f=raw_f, X=X, Xz=Xz, ica=ica, S=S, A=A,
                eog_proxy=eog_proxy, rows=rows, scores=scores, fs=fs)

def remove_artifacts(fit,
                     fp_ratio_thr=1.3,
                     corr_thr=0.50,
                     lfr_thr=1.8,
                     kurt_thr=20.0,
                     max_remove=2):
    """
    Select blink-like ICs using the given rule and reconstruct cleaned data
    OR fall back to Fp→F7/F8 regression when nothing to remove (or if rule
    selects all components).

    Args:
        fit (dict): output from fit_ICA(...)
    Returns:
        X_clean (np.ndarray): cleaned multichannel data (same shape as input X)
        blink_like (list): indices of removed components
    """
    X, S, A = fit["X"], fit["S"], fit["A"]
    rows = fit["rows"]

    # Selection Rule
    blink_like = []
    for comp, corr, lfr, k, fp_ratio in rows:
        is_frontal = fp_ratio >= fp_ratio_thr         # heavier on Fp than F7/F8
        blinky     = (abs(corr) >= corr_thr and lfr >= lfr_thr)
        spiky      = (k >= kurt_thr)
        if is_frontal and (blinky or spiky):
            blink_like.append(int(comp))

    blink_like = blink_like[:max_remove]
    print("Components to remove (conservative):", blink_like)

    # Reconstruct or FALL BACK to regression
    if len(blink_like) == 0 or len(blink_like) >= S.shape[1]:
        # Fallback: regress out Fp mean from F7/F8 (robust with 4ch)
        Fp = ((X[:, 0] + X[:, 1]) / 2.0)[:, None]
        Y  = X[:, 2:4]  # F7, F8
        beta = np.linalg.lstsq(Fp, Y, rcond=None)[0]
        Yc = Y - Fp @ beta
        X_clean = X.copy()
        X_clean[:, 2:4] = Yc
        note = "Fallback regression (Fp→F7/F8)"
    else:
        S_clean = S.copy()
        S_clean[:, blink_like] = 0
        # Undo z-scoring using original X stats
        X_mean = X.mean(axis=0, keepdims=True)
        X_std  = X.std(axis=0, keepdims=True) + 1e-9
        X_clean = (S_clean @ A.T) * X_std + X_mean
        note = f"ICA removed comps {blink_like}"

    return X_clean, blink_like, note


def rereference(x, method="car"):
    if method == "car":
        return x - x.mean(axis=1, keepdims=True)
    return x

def combine_channels(x30, method="mean"):
    """
    x30 shape: (samples, 4). Returns 1-D array (samples,)
    """
    if method == "mean":
        return x30.mean(axis=1)
    if method == "rms":
        return np.sqrt((x30**2).mean(axis=1))
    if method == "pca1":
        p = PCA(n_components=1)
        return p.fit_transform(x30).ravel()
    if method in {"ch1","ch2","ch3","ch4"}:
        idx = {"ch1":0,"ch2":1,"ch3":2,"ch4":3}[method]
        return x30[:, idx]
    raise ValueError("Unknown COMBINE_METHOD")

# ------------------ Task slicing + clean epoch ----------------------
def slice_task(x, start_s, dur_s, fs=FS):
    i0 = int(round(start_s * fs))
    i1 = i0 + int(round(dur_s * fs))
    i1 = min(i1, x.shape[0])
    return x[i0:i1]

def find_clean_epoch(x30, fs=FS, epoch_sec=CLEAN_EPOCH_SEC, thr=SPIKE_THRESHOLD_UV):
    """
    x30: (samples, 4) task segment (≈ 30 s). Return (epoch_samples, 4).
    We look for a contiguous window with no |amplitude| > thr on ANY channel.
    If none exists, choose the window with the fewest violations.
    """
    n = x30.shape[0]
    w = int(epoch_sec * fs)
    if n < w:
        return None

    bad = (np.abs(x30) > thr).any(axis=1).astype(int)  # 1 if any channel spikes
    c = np.cumsum(np.concatenate([[0], bad]))
    best_i, best_v = 0, math.inf
    for i in range(0, n - w + 1):
        v = c[i+w] - c[i]
        if v < best_v:
            best_v, best_i = v, i
            if v == 0:
                break
    return x30[best_i:best_i+w, :]

# --------------------- Welch PSD (median average) -------------------
def psd_median(x_epoch, fs=FS):
    """
    x_epoch: (samples, 4). Compute Welch PSD per channel, median-average across segments.
    Return freqs (Hz), PSD averaged across 4 channels (after channel-wise PSDs).
    """
    psds = []
    for ch in range(x_epoch.shape[1]):
        f, Pxx = welch(x_epoch[:, ch], fs=fs, window='hann',
                       nperseg=N_PERSEG, noverlap=N_OVERLP, detrend=False,
                       average='median', scaling='density')
        psds.append(Pxx)
    psd_mean = np.mean(psds, axis=0)
    return f, psd_mean

def slice_0p5_to_40(f, P):
    mask = (f >= 0.5) & (f <= 40.0)
    f2, P2 = f[mask], P[mask]
    return f2, P2

# --------------------- Band powers & TBR ----------------------------
@dataclass
class Bands:
    delta: tuple = (0.5, 4.0)
    theta: tuple = (4.0, 8.0)
    alpha: tuple = (8.0, 12.0)
    beta:  tuple = (12.0, 30.0)
    lgam:  tuple = (30.0, 40.0)

BANDS = Bands()

def band_power(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    if not np.any(m): return np.nan
    df = np.mean(np.diff(f[m])) if np.sum(m) > 1 else (hi - lo)
    return np.sum(P[m]) * df

def band_metrics(f, P):
    d = {
        "delta": band_power(f,P,*BANDS.delta),
        "theta": band_power(f,P,*BANDS.theta),
        "alpha": band_power(f,P,*BANDS.alpha),
        "beta":  band_power(f,P,*BANDS.beta),
        "low_gamma": band_power(f,P,*BANDS.lgam),
    }
    d["tbr"] = d["theta"] / d["beta"] if (d["theta"]>0 and d["beta"]>0) else np.nan
    d["total_power_0p5_40"] = band_power(f,P,0.5,40.0)
    return d

# -------------------------- Save processed fles -------------------------
def write_row_append(path, row_values, header=None):
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame([row_values])
    if (not os.path.exists(path)) and (header is not None):
        df.columns = header
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)

def collect_rows(path):
    return pd.read_csv(path) if os.path.exists(path) else None

# ---------------------------- Main Function -----------------------------
def process_all():
    epoch_dir = os.path.join(OUT_DIR, "epochs")
    psd_dir   = os.path.join(OUT_DIR, "psd80")
    band_dir  = os.path.join(OUT_DIR, "band_metrics")
    analysis_dir = os.path.join(OUT_DIR, "analysis")
    for p in [epoch_dir, psd_dir, band_dir, analysis_dir]:
        ensure_dir(p)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Found {len(files)} CSVs")
    for fp in files:
        cond, day = parse_type_and_day(fp)
        print(f"\n-> {os.path.basename(fp)}  [{cond}, day={day}]")
        try:
            raw = load_ganglion_csv(fp)             # (N,4)
            raw = apply_notch_and_bandpass(raw)     # filter
            fit = fit_ICA(raw)                  # fit ICA or regress
            raw_clean, _, _ = remove_artifacts(fit) # remove artifacts
            raw_car = rereference(raw_clean, REREF)           # re-ref
        except Exception as e:
            print(f"  ! Failed to load/preprocess: {e}")
            continue

        for task, (t0, dur) in TASKS.items():
            seg = slice_task(raw_car, t0, dur)
            if seg.shape[0] < CLEAN_EPOCH_SEC*FS:
                print(f"  - {task}: segment too short ({seg.shape[0]/FS:.1f}s), skipping")
                continue
            epoch = find_clean_epoch(seg)
            if epoch is None:
                print(f"  - {task}: no clean 15s window found, skipping")
                continue

            # single combined channel, 3000 samples in one row
            combined = combine_channels(epoch, COMBINE_METHOD)  # (3000,)
            epoch_path = os.path.join(epoch_dir, f"{cond}_{task}.csv")
            write_row_append(epoch_path, combined, header=[f"s{i}" for i in range(len(combined))])

            # PSD export (80-dim)
            for group, idxs in CHANNEL_GROUPS.items():
                f, P = psd_median(epoch[:, idxs])      # average across that group’s channels
                f2, P2 = slice_0p5_to_40(f, P)

                # save 80-D PSD for this group
                psd_path = os.path.join(psd_dir, f"{cond}_{task}__{group}.csv")
                write_row_append(psd_path, P2, header=[f"{fr:.1f}Hz" for fr in f2])

                # save band metrics for this group
                bm = band_metrics(f2, P2)
                meta_row = dict(day=day, **bm)
                band_path = os.path.join(band_dir, f"{cond}_{task}__{group}.csv")
                header = ["day","delta","theta","alpha","beta","low_gamma","tbr","total_power_0p5_40"]
                write_row_append(band_path, [meta_row.get(k, np.nan) for k in header], header=header)

            # band metrics
            bm = band_metrics(f2, P2)
            meta_row = dict(day=day, **bm)
            band_path = os.path.join(band_dir, f"{cond}_{task}.csv")
            # ensure columns order
            header = ["day","delta","theta","alpha","beta","low_gamma","tbr","total_power_0p5_40"]
            write_row_append(band_path, [meta_row.get(k, np.nan) for k in header], header=header)

    print("\n=== Pre-Processng Done. ===")

if __name__ == "__main__":
    process_all()