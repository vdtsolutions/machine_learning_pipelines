import pandas as pd
import numpy as np
import os
import re

# Optional imports
try:
    from scipy.fft import rfft, rfftfreq
    from scipy.stats import skew, kurtosis
    SCIPY_OK = True
except:
    SCIPY_OK = False

try:
    import pywt
    PYWT_OK = True
except:
    PYWT_OK = False

# ================= PATH CONFIG =================

SUBMATRIX_BASE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\PTTS"
MATCH_BASE     = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\defect_matcher\defect_match"
RESULTS_BASE   = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results"

PTTS = [1,2,3,4,5,6,7,8]
OUTPUT_CSV = "MASTER_ML_DATASET.csv"

# ================= PIPE METADATA =================
PIPE_OD_MM = 324
PIPE_WT_MM = 7.1
PIPE_INCH  = 12

# Column names
LABEL_COL      = "correct"
PRED_LEN_COL   = "pred_length"
TRUE_LEN_COL   = "actual_length_filtered"
LEN_DIFF_COL   = "length_diff"

# ============================================================

# ---------- Threshold crossings ----------
def threshold_crossings(signal, q):
    th = np.percentile(signal, q)
    binary = signal > th
    return int(np.sum(np.abs(np.diff(binary.astype(int)))))

# ---------- FFT ----------
def fft_features(signal):
    if not SCIPY_OK or len(signal) < 3:
        return {"fft_peak_freq": 0.0, "fft_peak_mag": 0.0, "fft_energy": 0.0}

    sig = signal - np.mean(signal)
    fft_vals = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), d=1)

    if len(fft_vals) < 2:
        return {"fft_peak_freq": 0.0, "fft_peak_mag": 0.0, "fft_energy": 0.0}

    peak_idx = np.argmax(fft_vals[1:]) + 1

    return {
        "fft_peak_freq": float(freqs[peak_idx]),
        "fft_peak_mag": float(fft_vals[peak_idx]),
        "fft_energy": float(np.sum(fft_vals**2) / len(sig))   # normalized
    }

# ---------- CWT ----------
def cwt_features(signal):
    if not PYWT_OK or len(signal) < 3:
        return {"cwt_energy": 0.0, "cwt_peak_scale": 0.0}

    scales = np.arange(1, 32)
    coeffs, _ = pywt.cwt(signal, scales, "mexh")

    energy = np.sum(coeffs**2) / len(signal)
    peak_scale = scales[np.argmax(np.mean(np.abs(coeffs), axis=1))]

    return {
        "cwt_energy": float(energy),
        "cwt_peak_scale": float(peak_scale)
    }

# ---------- Feature Extractor ----------
def extract_submatrix_features(path):

    try:
        df = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    except Exception as e:
        print(f"⚠️ Failed reading {path}: {e}")
        return None

    arr = df.to_numpy(dtype=str)

    # Extract numeric
    def extract_num(x):
        m = re.search(r"-?\d+\.?\d*", x)
        return float(m.group()) if m else np.nan

    mat = np.vectorize(extract_num)(arr)
    mat = np.nan_to_num(mat, nan=0.0)

    if mat.size == 0:
        return None

    # Axial signal (rows = axial)
    axial_signal = mat.mean(axis=1)
    original_len = len(axial_signal)

    # -------- Axial span (FIXED AXIS BUG) --------
    axial_active = np.any(mat > np.percentile(mat, 50), axis=1)
    axial_span = int(np.sum(axial_active))

    # -------- Threshold spans --------
    span_75 = int(np.sum(axial_signal > np.percentile(axial_signal, 75)))
    span_90 = int(np.sum(axial_signal > np.percentile(axial_signal, 90)))

    # -------- Threshold crossings --------
    cross_50 = threshold_crossings(axial_signal, 50)
    cross_75 = threshold_crossings(axial_signal, 75)
    cross_90 = threshold_crossings(axial_signal, 90)

    # -------- Gradient morphology --------
    if len(axial_signal) > 1:
        g = np.diff(axial_signal)
        grad_mean = float(np.mean(np.abs(g)))
        grad_std  = float(np.std(g))
    else:
        grad_mean = 0.0
        grad_std  = 0.0

    sk = float(skew(axial_signal)) if SCIPY_OK else 0.0
    ku = float(kurtosis(axial_signal)) if SCIPY_OK else 0.0

    # -------- Energy --------
    energy = float(np.sum(mat**2))
    energy_per_length = energy / max(axial_span, 1)

    # -------- FFT & CWT --------
    fft_f = fft_features(axial_signal)
    cwt_f = cwt_features(axial_signal)

    return {
        "axial_span_sub": axial_span,
        "mean_amp": float(mat.mean()),
        "max_amp": float(mat.max()),
        "std_amp": float(mat.std()),
        "energy": energy,
        "energy_per_length": energy_per_length,

        "cross_50": cross_50,
        "cross_75": cross_75,
        "cross_90": cross_90,
        "span_75": span_75,
        "span_90": span_90,

        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "skewness": sk,
        "kurtosis": ku,

        "is_short_defect": int(original_len < 8),

        **fft_f,
        **cwt_f
    }

# ============================================================

all_rows = []

# ================= LOOP ALL PTTs =================

for ptt in PTTS:
    print(f"\n========== PROCESSING PTT {ptt} ==========")

    match_path   = os.path.join(MATCH_BASE, f"results_matching_PTT_{ptt}.csv")
    results_path = os.path.join(RESULTS_BASE, f"PTT_{ptt}_RESULTS.csv")
    submat_dir   = os.path.join(SUBMATRIX_BASE, f"PTT_{ptt}")

    if not os.path.exists(match_path) or not os.path.exists(results_path) or not os.path.exists(submat_dir):
        print(f"❌ Missing files for PTT {ptt}")
        continue

    match_df   = pd.read_csv(match_path)
    results_df = pd.read_csv(results_path)

    # Normalize labels
    match_df[LABEL_COL] = match_df[LABEL_COL].astype(str).str.strip().str.upper()
    match_df = match_df[match_df[LABEL_COL].isin(["YES", "NO"])]

    # ID conversion
    match_df["id"]   = pd.to_numeric(match_df["id"], errors="coerce")
    results_df["id"] = pd.to_numeric(results_df["id"], errors="coerce")

    match_df   = match_df.dropna(subset=["id"])
    results_df = results_df.dropna(subset=["id"])

    match_df["id"]   = match_df["id"].astype(int)
    results_df["id"] = results_df["id"].astype(int)

    # Merge
    df = match_df.merge(results_df, on="id", how="left")
    df["correct"] = df[LABEL_COL].map({"YES": 1, "NO": 0})

    print(df["correct"].value_counts())

    # Loop defects
    for _, row in df.iterrows():
        defect_id = int(row["id"])

        files = [f for f in os.listdir(submat_dir) if f.startswith(f"submatrix_ptt-{ptt}({defect_id},")]
        if not files:
            continue

        sub_path = os.path.join(submat_dir, files[0])
        f = extract_submatrix_features(sub_path)
        if f is None:
            continue

        all_rows.append({

            # Identity
            "ptt": ptt,
            "id": defect_id,

            # Length info
            "pred_length": row.get(PRED_LEN_COL, np.nan),
            "true_length": row.get(TRUE_LEN_COL, np.nan),
            "length_diff": row.get(LEN_DIFF_COL, np.nan),

            # Pipe metadata
            "pipe_od_mm": PIPE_OD_MM,
            "pipe_wall_thickness_mm": PIPE_WT_MM,
            "pipe_nominal_inch": PIPE_INCH,

            # Matching meta
            "distance_mm": row.get("distance_mm", np.nan),
            "orientation_match": row.get("orientation", np.nan),

            # PTT results meta
            "start_index": row.get("start_index", np.nan),
            "end_index": row.get("end_index", np.nan),
            "start_sensor": row.get("start_sensor", np.nan),
            "end_sensor": row.get("end_sensor", np.nan),
            "absolute_distance": row.get("absolute_distance", np.nan),
            "upstream": row.get("upstream", np.nan),
            "speed": row.get("speed", np.nan),
            "Min_Val": row.get("Min_Val", np.nan),
            "Max_Val": row.get("Max_Val", np.nan),
            "defect_type": row.get("defect_type", np.nan),
            "dimension_classification": row.get("dimension_classification", np.nan),

            # Submatrix features
            **f,

            # Label
            "correct": row["correct"]
        })

# Save dataset
master_df = pd.DataFrame(all_rows)
master_df.to_csv(OUTPUT_CSV, index=False)

print("\n================ DONE ================")
print("Saved:", OUTPUT_CSV)
print("Shape:", master_df.shape)
print("\nFinal Label Distribution:")
print(master_df["correct"].value_counts())
