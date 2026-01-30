import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report

# ================= CONFIG =================
MAX_SENSOR_COUNT = 144
OUTLIER_THRESHOLD_RATIO = 0.6
TOL_MM = 11

# ================= LOAD DATA =================
df = pd.read_csv(r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET_V2.csv")
df = df.dropna(subset=["pred_length", "true_length"])

# ================= SENSOR SPAN + OUTLIERS =================
df["sensor_span"] = df["end_sensor"] - df["start_sensor"] + 1
df["sensor_span_ratio"] = df["sensor_span"] / MAX_SENSOR_COUNT
df["outlier"] = (df["sensor_span_ratio"] > OUTLIER_THRESHOLD_RATIO).astype(int)

# ================= AXIAL INDEX SPAN =================
df["axial_index_span"] = df["end_index"] - df["start_index"]

print("\n================ DATA SUMMARY ================")
print("Total rows:", len(df))
print("Total outliers:", df["outlier"].sum())

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

# ================= TRAIN / TEST SPLIT =================
TRAIN_PTTS = [1,2,3,4,5,6]
TEST_PTTS  = [7,8]

train_df = df[df["ptt"].isin(TRAIN_PTTS)]
test_df  = df[df["ptt"].isin(TEST_PTTS)]

# ================= REMOVE OUTLIERS FROM TRAIN =================
train_df = train_df[train_df["outlier"] == 0]
print("\nTraining samples after outlier removal:", train_df.shape)

X_train = train_df[FEATURES]
y_train = train_df["true_length"] - train_df["pred_length"]

# ================= TRAIN MODEL =================
reg = XGBRegressor(
    n_estimators=1500,
    max_depth=15,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
reg.fit(X_train, y_train)

# ==============================================================
# FUNCTION TO APPLY MODEL TO ANY DF
# ==============================================================

def apply_model(df_in):
    df_in = df_in.copy()
    X = df_in[FEATURES]

    df_in["pred_corrected"] = df_in["pred_length"]
    corrections = reg.predict(X)

    # Safety clamp ±30%
    corrections = np.clip(corrections,
                          -0.3 * df_in["pred_length"],
                           0.3 * df_in["pred_length"])

    # Apply only to non-outliers
    mask = df_in["outlier"] == 0
    df_in.loc[mask, "pred_corrected"] = df_in.loc[mask, "pred_length"] + corrections[mask]

    # Tiny defect gating
    tiny_mask = df_in["pred_length"] < 10
    df_in.loc[tiny_mask, "pred_corrected"] = df_in.loc[tiny_mask, "pred_length"]

    return df_in

# ================= APPLY TO TEST =================
test_df = apply_model(test_df)

# ================= APPLY TO ALL DATA (CONSOLIDATED) =================
full_df = apply_model(df)

# ==============================================================
# METRICS FUNCTION
# ==============================================================

def pct_within_tol(y_true, y_pred, tol=0.10):
    return np.mean(np.abs(y_true - y_pred) / y_true < tol) * 100

def compute_accuracy_flags(df_eval):
    df_eval = df_eval.copy()
    df_eval["prev_correct"]  = (np.abs(df_eval["true_length"] - df_eval["pred_length"]) < TOL_MM).astype(int)
    df_eval["after_correct"] = (np.abs(df_eval["true_length"] - df_eval["pred_corrected"]) < TOL_MM).astype(int)
    return df_eval

# ================= EVAL TEST DATA =================
eval_test = test_df[test_df["outlier"] == 0].copy()
eval_test = compute_accuracy_flags(eval_test)

print("\n================ TEST GLOBAL REGRESSION METRICS ================")
print("Baseline MAE:", mean_absolute_error(eval_test["true_length"], eval_test["pred_length"]))
print("ML MAE:", mean_absolute_error(eval_test["true_length"], eval_test["pred_corrected"]))
print("Baseline R2:", r2_score(eval_test["true_length"], eval_test["pred_length"]))
print("ML R2:", r2_score(eval_test["true_length"], eval_test["pred_corrected"]))
print("% within ±10% Baseline:", pct_within_tol(eval_test["true_length"], eval_test["pred_length"]))
print("% within ±10% ML:", pct_within_tol(eval_test["true_length"], eval_test["pred_corrected"]))

print("\n================ GLOBAL CONFUSION MATRIX (TEST) ================")
cm = confusion_matrix(eval_test["prev_correct"], eval_test["after_correct"])
print(cm)
print(classification_report(eval_test["prev_correct"], eval_test["after_correct"], digits=4))

# ==============================================================
# PTT-WISE ACCURACY FOR ALL PTTs
# ==============================================================

acc_rows = []

print("\n================ PTT-WISE ACCURACY ALL PTTs ================")

for ptt, g in full_df.groupby("ptt"):
    g = g[g["outlier"] == 0]
    if len(g) == 0:
        continue

    g = compute_accuracy_flags(g)

    prev_acc  = g["prev_correct"].mean()
    after_acc = g["after_correct"].mean()

    print(f"PTT {ptt}: Before={prev_acc:.4f}, After={after_acc:.4f}, N={len(g)}")

    acc_rows.append({
        "ptt_id": ptt,
        "baseline_accuracy": prev_acc,
        "ml_accuracy": after_acc,
        "n_samples": len(g)
    })

acc_df = pd.DataFrame(acc_rows)

# ==============================================================
# SAVE TEST CSV (UNCHANGED FORMAT)
# ==============================================================

test_out = test_df[["ptt","id","true_length","pred_length","pred_corrected","outlier"]].copy()
test_out.columns = ["ptt_id","id","actual_length","pred_length","model_pred_length","outlier"]

test_out["prev_correct"]  = (np.abs(test_out["actual_length"] - test_out["pred_length"]) < TOL_MM).astype(int)
test_out["after_model_correct"] = (np.abs(test_out["actual_length"] - test_out["model_pred_length"]) < TOL_MM).astype(int)
test_out["fixed_by_model"]  = ((test_out["prev_correct"]==0) & (test_out["after_model_correct"]==1)).astype(int)
test_out["ruined_by_model"] = ((test_out["prev_correct"]==1) & (test_out["after_model_correct"]==0)).astype(int)

TEST_OUT_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\TEST_PREDICTIONS_FINAL.csv"
test_out.to_csv(TEST_OUT_PATH, index=False)

# ==============================================================
# SAVE CONSOLIDATED CSV (ALL PTTs)
# ==============================================================

full_out = full_df[["ptt","id","true_length","pred_length","pred_corrected","outlier"]].copy()
full_out.columns = ["ptt_id","id","actual_length","pred_length","model_pred_length","outlier"]

full_out["prev_correct"]  = (np.abs(full_out["actual_length"] - full_out["pred_length"]) < TOL_MM).astype(int)
full_out["after_model_correct"] = (np.abs(full_out["actual_length"] - full_out["model_pred_length"]) < TOL_MM).astype(int)
full_out["fixed_by_model"]  = ((full_out["prev_correct"]==0) & (full_out["after_model_correct"]==1)).astype(int)
full_out["ruined_by_model"] = ((full_out["prev_correct"]==1) & (full_out["after_model_correct"]==0)).astype(int)

FULL_OUT_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\CONSOLIDATED_ALL_PTT_PREDICTIONS.csv"
full_out.to_csv(FULL_OUT_PATH, index=False)

# ==============================================================
# SAVE PTT ACCURACY SUMMARY CSV
# ==============================================================

ACC_OUT_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\PTT_ACCURACY_SUMMARY.csv"
acc_df.to_csv(ACC_OUT_PATH, index=False)

# ==============================================================
# FINAL PRINT
# ==============================================================

print("\n================ FILES SAVED ================")
print("TEST CSV:", TEST_OUT_PATH)
print("CONSOLIDATED CSV:", FULL_OUT_PATH)
print("PTT ACCURACY CSV:", ACC_OUT_PATH)
