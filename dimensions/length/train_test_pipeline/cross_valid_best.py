import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ================= CONFIG =================
MAX_SENSOR_COUNT = 144
OUTLIER_THRESHOLD_RATIO = 0.6
TOL_MM = 12
N_TRIALS = 20

DATA_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET_V2.csv"
OUT_DIR = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\AUTO_CV"
os.makedirs(OUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["pred_length", "true_length"])

# ================= SENSOR SPAN =================
df["sensor_span"] = df["end_sensor"] - df["start_sensor"] + 1
df["sensor_span_ratio"] = df["sensor_span"] / MAX_SENSOR_COUNT

# ================= AXIAL INDEX SPAN =================
df["axial_index_span"] = df["end_index"] - df["start_index"]

# ================= FEATURES =================
FEATURES = [
    "pred_length",
    "axial_index_span",
    "axial_span_sub",
    "span_75", "span_90",
    "grad_mean", "grad_std",
    "energy_per_length",
    "fft_energy",
    "cwt_energy"
]
FEATURES = [f for f in FEATURES if f in df.columns]
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

# ============================================================
# OUTLIER DETECTION FUNCTION
# ============================================================
def detect_outliers(df):
    df = df.copy()

    # Rule 1: Sensor span outlier
    df["outlier_sensor"] = df["sensor_span_ratio"] > OUTLIER_THRESHOLD_RATIO

    # Rule 2: Length inconsistency outlier
    df["outlier_length"] = df["true_length"] > 2 * (df["pred_length"] + TOL_MM)

    # Label outlier type
    def label_outlier(row):
        if row["outlier_sensor"] and row["outlier_length"]:
            return "both"
        elif row["outlier_sensor"]:
            return "sensor"
        elif row["outlier_length"]:
            return "length"
        else:
            return "none"

    df["outlier_type"] = df.apply(label_outlier, axis=1)

    # Unified flag
    df["outlier"] = ((df["outlier_sensor"]) | (df["outlier_length"])).astype(int)

    return df

df = detect_outliers(df)

# ================= DATA SUMMARY =================
print("\n================ DATA SUMMARY ================")
print(f"Total rows: {len(df)}")
print(f"Total outliers: {df['outlier'].sum()}")
print(f"Sensor outliers: {df['outlier_sensor'].sum()}")
print(f"Length outliers: {df['outlier_length'].sum()}")
print(f"Both-rule outliers: {(df['outlier_type']=='both').sum()}")
print("Features used:", FEATURES)

# ================= SAVE OUTLIER LIST CSV =================
outlier_csv = df[df["outlier"] == 1][["ptt", "id", "outlier_type"]].copy()
outlier_csv.columns = ["ptt_id", "id", "reason"]
OUTLIER_PATH = os.path.join(OUT_DIR, "OUTLIERS.csv")
outlier_csv.to_csv(OUTLIER_PATH, index=False)
print(f"\nSaved OUTLIERS.csv -> {OUTLIER_PATH}")

# ================= TARGET =================
df["target_residual"] = df["true_length"] - df["pred_length"]

# ================= METRICS =================
def pct_within_tol(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) < TOL_MM)

# ================= PTT LIST =================
PTTS = sorted(df["ptt"].unique())

# ================= RANDOM PARAM GENERATOR =================
def sample_params():
    return {
        "n_estimators": np.random.randint(200, 1200),
        "max_depth": np.random.randint(3, 10),
        "learning_rate": np.random.uniform(0.02, 0.15),
        "subsample": np.random.uniform(0.7, 1.0),
        "colsample_bytree": np.random.uniform(0.7, 1.0),
        "random_state": 42
    }

# ================= TRIAL LOG STORAGE =================
trial_log_rows = []
trial_results = []

# ================= CROSS-VALIDATION (NO OUTLIERS) =================
print("\n================ STARTING HYPERPARAM SEARCH ================")

for trial in range(N_TRIALS):
    params = sample_params()
    print(f"\n---------------- TRIAL {trial+1}/{N_TRIALS} ----------------")
    print("Params:", params)

    fold_ml_scores = []
    fold_baseline_scores = []

    for val_ptt in PTTS:
        train_df = df[(df["ptt"] != val_ptt) & (df["outlier"] == 0)]
        val_df   = df[(df["ptt"] == val_ptt) & (df["outlier"] == 0)]

        if len(val_df) < 10:
            continue

        X_train = train_df[FEATURES]
        y_train = train_df["target_residual"]
        X_val = val_df[FEATURES]
        y_val = val_df["true_length"]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        pred_corrected = val_df["pred_length"] + model.predict(X_val)

        baseline_acc = pct_within_tol(y_val, val_df["pred_length"])
        ml_acc = pct_within_tol(y_val, pred_corrected)

        fold_baseline_scores.append(baseline_acc)
        fold_ml_scores.append(ml_acc)

        print(f"PTT {val_ptt} | Baseline={baseline_acc:.4f} | ML={ml_acc:.4f} | N={len(val_df)}")

    mean_baseline = np.mean(fold_baseline_scores)
    mean_ml = np.mean(fold_ml_scores)

    print(f">>> TRIAL {trial+1} BASELINE ACC = {mean_baseline:.4f}")
    print(f">>> TRIAL {trial+1} ML ACC = {mean_ml:.4f}")

    trial_results.append((mean_ml, params))
    trial_log_rows.append({
        "trial_id": trial+1,
        "baseline_cv_accuracy_clean": mean_baseline,
        "ml_cv_accuracy_clean": mean_ml,
        "params": str(params)
    })

# ================= SAVE TRIAL LOG =================
trial_log_df = pd.DataFrame(trial_log_rows)
TRIAL_LOG_PATH = os.path.join(OUT_DIR, "TRIAL_ACCURACY_LOG.csv")
trial_log_df.to_csv(TRIAL_LOG_PATH, index=False)
print(f"\nSaved TRIAL_ACCURACY_LOG.csv -> {TRIAL_LOG_PATH}")

# ================= PICK BEST MODEL =================
trial_results.sort(key=lambda x: x[0], reverse=True)
best_acc, best_params = trial_results[0]

print("\n================ BEST MODEL SELECTED ================")
print(f"Best CV Accuracy = {best_acc}")
print("Best Params:", best_params)

# ================= TRAIN BEST MODEL ON ALL CLEAN DATA =================
train_all = df[df["outlier"] == 0]
X_train = train_all[FEATURES]
y_train = train_all["target_residual"]

best_model = XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

print("\nModel trained on ALL CLEAN data")

# ================= APPLY MODEL =================
def apply_model(df_in):
    df_in = df_in.copy()
    X = df_in[FEATURES]

    corrections = best_model.predict(X)
    corrections = np.clip(corrections, -0.3 * df_in["pred_length"], 0.3 * df_in["pred_length"])

    df_in["pred_corrected"] = df_in["pred_length"]
    mask = df_in["outlier"] == 0
    df_in.loc[mask, "pred_corrected"] = df_in.loc[mask, "pred_length"] + corrections[mask]

    tiny = df_in["pred_length"] < 10
    df_in.loc[tiny, "pred_corrected"] = df_in.loc[tiny, "pred_length"]

    return df_in

full_df = apply_model(df)

# ================= CONSOLIDATED CSV =================
full_out = full_df[["ptt","id","true_length","pred_length","pred_corrected","outlier","outlier_type"]].copy()
full_out.columns = ["ptt_id","id","actual_length","pred_length","model_pred_length","outlier","outlier_type"]

# Accuracy only for clean
mask = full_out["outlier"] == 0

full_out["prev_correct"] = 0
full_out["after_model_correct"] = 0

full_out.loc[mask, "prev_correct"] = (np.abs(full_out.loc[mask, "actual_length"] - full_out.loc[mask, "pred_length"]) < TOL_MM).astype(int)
full_out.loc[mask, "after_model_correct"] = (np.abs(full_out.loc[mask, "actual_length"] - full_out.loc[mask, "model_pred_length"]) < TOL_MM).astype(int)

full_out["fixed_by_model"]  = ((full_out["prev_correct"]==0) & (full_out["after_model_correct"]==1)).astype(int)
full_out["ruined_by_model"] = ((full_out["prev_correct"]==1) & (full_out["after_model_correct"]==0)).astype(int)

FULL_OUT_PATH = os.path.join(OUT_DIR, "CONSOLIDATED_ALL_PTT_PREDICTIONS.csv")
full_out.to_csv(FULL_OUT_PATH, index=False)
print(f"\nSaved CONSOLIDATED CSV -> {FULL_OUT_PATH}")

# ================= GLOBAL ACCURACY REPORT =================
clean = full_out[full_out.outlier == 0]
all_df = full_out.copy()

print("\n================ GLOBAL ACCURACY REPORT ================")
print(f"Baseline Accuracy (CLEAN): {clean['prev_correct'].mean():.4f}")
print(f"ML Accuracy (CLEAN): {clean['after_model_correct'].mean():.4f}")
print(f"Baseline Accuracy (ALL): {(np.abs(all_df['actual_length'] - all_df['pred_length']) < TOL_MM).mean():.4f}")
print(f"ML Accuracy (ALL): {(np.abs(all_df['actual_length'] - all_df['model_pred_length']) < TOL_MM).mean():.4f}")

# ================= PTT ACCURACY SUMMARY =================
acc_rows = []
print("\n================ PTT ACCURACY SUMMARY ================")

for ptt, g in full_out.groupby("ptt_id"):
    clean_g = g[g["outlier"] == 0]
    if len(clean_g) == 0:
        continue

    prev_clean = clean_g["prev_correct"].mean()
    after_clean = clean_g["after_model_correct"].mean()

    prev_all = (np.abs(g["actual_length"] - g["pred_length"]) < TOL_MM).mean()
    after_all = (np.abs(g["actual_length"] - g["model_pred_length"]) < TOL_MM).mean()

    print(f"PTT {ptt}: CLEAN Before={prev_clean:.4f}, After={after_clean:.4f} | ALL Before={prev_all:.4f}, After={after_all:.4f} | N={len(g)}")

    acc_rows.append({
        "ptt_id": ptt,
        "baseline_accuracy_clean": prev_clean,
        "ml_accuracy_clean": after_clean,
        "baseline_accuracy_all": prev_all,
        "ml_accuracy_all": after_all,
        "n_samples": len(g)
    })

acc_df = pd.DataFrame(acc_rows)
ACC_OUT_PATH = os.path.join(OUT_DIR, "PTT_ACCURACY_SUMMARY.csv")
acc_df.to_csv(ACC_OUT_PATH, index=False)

# ================= FINAL PRINT =================
print("\n================ FILES SAVED ================")
print("TRIAL ACCURACY LOG:", TRIAL_LOG_PATH)
print("OUTLIERS CSV:", OUTLIER_PATH)
print("CONSOLIDATED CSV:", FULL_OUT_PATH)
print("PTT ACCURACY CSV:", ACC_OUT_PATH)
