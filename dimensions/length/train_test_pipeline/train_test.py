import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error

# ================= LOAD DATA =================
df = pd.read_csv(r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET.csv")

# Drop bad rows
df = df.dropna(subset=["pred_length", "true_length", "correct"])

# ================= FEATURE SELECTION =================

FEATURES = [
    # baseline
    "pred_length",

    # morphology
    "axial_span_sub",
    "span_75", "span_90",
    "grad_mean", "grad_std",
    "skewness", "kurtosis",
    "energy_per_length",

    # spectral
    "fft_peak_freq", "fft_peak_mag", "fft_energy",
    "cwt_energy", "cwt_peak_scale",

    # tool physics
    "speed",
    "orientation_match",

    # pipe metadata
    "pipe_od_mm", "pipe_wall_thickness_mm",

    # STEP 4 EXTRA IMPORTANT FEATURES
    "start_index", "end_index",
    "Min_Val", "Max_Val",
    "absolute_distance", "upstream"
]

# Keep only existing columns
FEATURES = [f for f in FEATURES if f in df.columns]

# Convert to numeric
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df[FEATURES] = df[FEATURES].fillna(0)

# ================= TRAIN / TEST SPLIT BY PTT =================
TRAIN_PTTS = [1,2,3,4,5,6]
TEST_PTTS  = [7,8]

train_df = df[df["ptt"].isin(TRAIN_PTTS)]
test_df  = df[df["ptt"].isin(TEST_PTTS)]

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

# ================= STAGE 1: CLASSIFIER =================
y_train_cls = 1 - train_df["correct"]   # 1 = WRONG
y_test_cls  = 1 - test_df["correct"]

cls = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

cls.fit(X_train, y_train_cls)

print("\n===== CLASSIFIER PERFORMANCE =====")
print(classification_report(y_test_cls, cls.predict(X_test)))

# ================= STAGE 2: STRONGER REGRESSOR =================
train_no = train_df[train_df["correct"] == 0]   # WRONG ONLY

X_train_reg = train_no[FEATURES]
y_train_reg = train_no["true_length"] - train_no["pred_length"]

reg = XGBRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)

reg.fit(X_train_reg, y_train_reg)

# ================= APPLY CORRECTION ON TEST =================
test_df = test_df.copy()
test_df["pred_corrected"] = test_df["pred_length"]

# Conservative gating
proba_wrong = cls.predict_proba(X_test)[:,1]
wrong_pred_mask = proba_wrong > 0.7

# Predict corrections
corrections = reg.predict(X_test)

# Apply correction only to confident wrong
test_df.loc[wrong_pred_mask, "pred_corrected"] = (
    test_df.loc[wrong_pred_mask, "pred_length"] + corrections[wrong_pred_mask]
)

# ================= EVALUATION =================

def pct_within_tol(y_true, y_pred, tol=0.10):
    return np.mean(np.abs(y_true - y_pred) / y_true < tol) * 100

baseline_mae = mean_absolute_error(test_df["true_length"], test_df["pred_length"])
ml_mae       = mean_absolute_error(test_df["true_length"], test_df["pred_corrected"])

print("\n===== LENGTH ACCURACY RESULTS =====")
print("Baseline MAE:", baseline_mae)
print("ML Corrected MAE:", ml_mae)
print("Improvement %:", (baseline_mae - ml_mae) / baseline_mae * 100)

print("\n% within ±10% tolerance (Baseline):",
      pct_within_tol(test_df["true_length"], test_df["pred_length"]))

print("% within ±10% tolerance (ML):",
      pct_within_tol(test_df["true_length"], test_df["pred_corrected"]))

# ================= SAFETY CHECK =================

baseline_correct = np.abs(test_df["true_length"] - test_df["pred_length"]) < 11
ml_correct       = np.abs(test_df["true_length"] - test_df["pred_corrected"]) < 11

ruined = np.sum((baseline_correct == True) & (ml_correct == False))
fixed  = np.sum((baseline_correct == False) & (ml_correct == True))

print("\n===== SAFETY METRICS =====")
print("Correct ruined:", ruined)
print("Wrong fixed:", fixed)

# ================= SAVE TEST CSV WITH FLAGS =================

# ================= SAVE TEST CSV WITH FLAGS =================

# Include PTT ID + ID
out_df = test_df[["ptt", "id", "true_length", "pred_length", "pred_corrected"]].copy()
out_df.columns = ["ptt_id", "id", "actual_length", "pred_length", "model_pred_length"]

# Add correctness flags
out_df["prev_correct"] = baseline_correct.astype(int)
out_df["after_model_correct"] = ml_correct.astype(int)

# Improvement flags
out_df["fixed_by_model"] = ((baseline_correct == False) & (ml_correct == True)).astype(int)
out_df["ruined_by_model"] = ((baseline_correct == True) & (ml_correct == False)).astype(int)

# Save CSV
out_path = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\TEST_PREDICTIONS.csv"
out_df.to_csv(out_path, index=False)

print("\nSaved test predictions to:", out_path)

