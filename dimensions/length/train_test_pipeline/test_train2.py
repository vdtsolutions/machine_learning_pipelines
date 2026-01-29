import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report

# ================= CONFIG =================
MAX_SENSOR_COUNT = 144
OUTLIER_SENSOR_RATIO = 0.6   # >60% sensors = weld-like
TOL_MM = 11

TRAIN_PTTS = [1,2,3,4,5,6]
TEST_PTTS  = [7,8]

# ================= LOAD DATA =================
df = pd.read_csv(r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET.csv")
df = df.dropna(subset=["pred_length", "true_length"])
df = df[df["pred_length"] > 0]

# ================= SENSOR OUTLIERS =================
df["sensor_span"] = df["end_sensor"] - df["start_sensor"] + 1
df["sensor_span_ratio"] = df["sensor_span"] / MAX_SENSOR_COUNT
df["outlier"] = (df["sensor_span_ratio"] > OUTLIER_SENSOR_RATIO).astype(int)

# ================= AXIAL PHYSICS FEATURE =================
df["axial_index_span"] = df["end_index"] - df["start_index"]

# ================= DATA SUMMARY =================
print("\n================ DATA SUMMARY ================")
print("Total rows:", len(df))
print("Total outliers:", df["outlier"].sum())

print("\n================ OUTLIER REPORT PER PTT ================")
for ptt in sorted(df["ptt"].unique()):
    ids = df[(df["ptt"]==ptt) & (df["outlier"]==1)]["id"].tolist()
    print(f"PTT {ptt}: Outliers = {len(ids)}")
    if ids:
        print("  IDs:", ids)

# ================= FEATURES =================
FEATURES = [
    "pred_length",
    "axial_index_span",
    "axial_span_sub","span_75","span_90",
    "grad_mean","grad_std","skewness","kurtosis","energy_per_length",
    "fft_peak_freq","fft_peak_mag","fft_energy","cwt_energy","cwt_peak_scale",
    "speed","orientation_match",
    "pipe_od_mm","pipe_wall_thickness_mm",
    "Min_Val","Max_Val","absolute_distance","upstream"
]

FEATURES = [f for f in FEATURES if f in df.columns]
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

# ================= SPLIT TRAIN / TEST =================
train_df = df[df["ptt"].isin(TRAIN_PTTS)]
test_df  = df[df["ptt"].isin(TEST_PTTS)]

# Remove outliers from training
train_df = train_df[train_df["outlier"] == 0]

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

# ================= TARGET = RESIDUAL =================
y_train = train_df["true_length"] - train_df["pred_length"]

# ================= TRAIN MODEL (STRONG REGULARIZATION) =================
reg = XGBRegressor(
    n_estimators=2500,
    max_depth=4,
    learning_rate=0.015,
    subsample=0.6,
    colsample_bytree=0.6,
    min_child_weight=20,
    gamma=1.0,
    reg_lambda=2.0,
    random_state=42
)
reg.fit(X_train, y_train)

# ================= APPLY MODEL =================
test_df = test_df.copy()
test_df["pred_corrected"] = test_df["pred_length"]

corrections = reg.predict(X_test)

# Safety clamp (max ±25%)
corrections = np.clip(corrections,
                      -0.25 * test_df["pred_length"],
                       0.25 * test_df["pred_length"])

# Apply only to non-outliers
test_df.loc[test_df["outlier"]==0, "pred_corrected"] = (
    test_df.loc[test_df["outlier"]==0, "pred_length"] + corrections[test_df["outlier"]==0]
)

# ================= EVAL ONLY NON-OUTLIERS =================
eval_df = test_df[test_df["outlier"]==0].copy()

# ================= REGRESSION METRICS =================
def pct_within_tol(y_true, y_pred, tol=0.10):
    return np.mean(np.abs(y_true - y_pred) / y_true < tol) * 100

baseline_mae = mean_absolute_error(eval_df["true_length"], eval_df["pred_length"])
ml_mae       = mean_absolute_error(eval_df["true_length"], eval_df["pred_corrected"])
baseline_r2  = r2_score(eval_df["true_length"], eval_df["pred_length"])
ml_r2        = r2_score(eval_df["true_length"], eval_df["pred_corrected"])

print("\n================ GLOBAL REGRESSION METRICS ================")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"ML MAE: {ml_mae:.4f}")
print(f"Improvement %: {(baseline_mae-ml_mae)/baseline_mae*100:.2f}%")
print(f"Baseline R2: {baseline_r2:.4f}")
print(f"ML R2: {ml_r2:.4f}")
print(f"% within ±10% (Baseline): {pct_within_tol(eval_df['true_length'], eval_df['pred_length']):.2f}")
print(f"% within ±10% (ML): {pct_within_tol(eval_df['true_length'], eval_df['pred_corrected']):.2f}")

# ================= CORRECTNESS FLAGS =================
eval_df["prev_correct"]  = (np.abs(eval_df["true_length"] - eval_df["pred_length"]) < TOL_MM).astype(int)
eval_df["after_correct"] = (np.abs(eval_df["true_length"] - eval_df["pred_corrected"]) < TOL_MM).astype(int)

# ================= GLOBAL CONFUSION MATRIX =================
print("\n================ GLOBAL CONFUSION MATRIX ================")
cm_global = confusion_matrix(eval_df["prev_correct"], eval_df["after_correct"])
print(cm_global)
print("\nGLOBAL CLASSIFICATION REPORT:")
print(classification_report(eval_df["prev_correct"], eval_df["after_correct"], digits=4))

# ================= PTT-WISE REPORT =================
ptt_acc = {}

print("\n================ PTT-WISE CONFUSION MATRICES ================")

for ptt in TEST_PTTS:
    ptt_df = eval_df[eval_df["ptt"] == ptt]
    if len(ptt_df)==0:
        continue

    prev = ptt_df["prev_correct"]
    after = ptt_df["after_correct"]
    cm = confusion_matrix(prev, after)

    print(f"\n---------- PTT {ptt} ----------")
    print("Samples (non-outliers):", len(ptt_df))
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(prev, after, digits=4))

    prev_acc = prev.mean()
    after_acc = after.mean()
    print(f"Baseline Accuracy: {prev_acc:.4f}")
    print(f"ML Accuracy: {after_acc:.4f}")

    ptt_acc[ptt] = (prev_acc, after_acc)

# ================= SAVE CSV =================
out_df = test_df[["ptt","id","true_length","pred_length","pred_corrected","outlier"]].copy()
out_df.columns = ["ptt_id","id","actual_length","pred_length","model_pred_length","outlier"]

# correctness flags
out_df["prev_correct"] = np.where(out_df["outlier"]==0,
                                  (np.abs(out_df["actual_length"]-out_df["pred_length"])<TOL_MM).astype(int),
                                  np.nan)

out_df["after_model_correct"] = np.where(out_df["outlier"]==0,
                                         (np.abs(out_df["actual_length"]-out_df["model_pred_length"])<TOL_MM).astype(int),
                                         np.nan)

out_df["fixed_by_model"]  = ((out_df["prev_correct"]==0) & (out_df["after_model_correct"]==1)).astype(int)
out_df["ruined_by_model"] = ((out_df["prev_correct"]==1) & (out_df["after_model_correct"]==0)).astype(int)

# attach PTT accuracy
out_df["ptt_baseline_accuracy"] = out_df["ptt_id"].map(lambda x: ptt_acc.get(x,(np.nan,np.nan))[0])
out_df["ptt_ml_accuracy"]       = out_df["ptt_id"].map(lambda x: ptt_acc.get(x,(np.nan,np.nan))[1])

out_path = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\FINAL_TEST_PREDICTIONS.csv"
out_df.to_csv(out_path, index=False)

print("\n================ CSV SAVED ================")
print(out_path)
