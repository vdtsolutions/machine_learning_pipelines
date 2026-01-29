import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ================= CONFIG =================
DATA_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET.csv"
OUT_DIR = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline"
TOL_MM = 11

TRAIN_PTTS = [1,2,3,4,5,6]
TEST_PTTS  = [7,8]

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["pred_length", "true_length"])

# ================= FEATURES =================
FEATURES = [
    "pred_length",
    "axial_index_span",
    "axial_span_sub",
    "span_75", "span_90",
    "grad_mean", "grad_std",
    "energy_per_length",

    # optional spectral (low weight)
    "fft_energy",
    "cwt_energy"
]


FEATURES = [f for f in FEATURES if f in df.columns]

df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

# ================= TRAIN / TEST SPLIT =================
train_df = df[df["ptt"].isin(TRAIN_PTTS)]
test_df  = df[df["ptt"].isin(TEST_PTTS)]

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

y_train = train_df["true_length"] - train_df["pred_length"]
y_test  = test_df["true_length"] - test_df["pred_length"]

# ================= TRAIN BASELINE MODEL =================
reg = XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
reg.fit(X_train, y_train)

# ================= BASELINE EVAL =================
baseline_pred = test_df["pred_length"] + reg.predict(X_test)
baseline_mae = mean_absolute_error(test_df["true_length"], baseline_pred)
baseline_r2  = r2_score(test_df["true_length"], baseline_pred)

print("\n================ BASELINE METRICS ================")
print(f"MAE: {baseline_mae:.4f}")
print(f"R2 : {baseline_r2:.4f}")

# ================= PERMUTATION IMPORTANCE =================
print("\n================ PERMUTATION IMPORTANCE ================")

perm = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42)
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

print(imp_df)

# Save importance CSV
imp_path = OUT_DIR + r"\feature_importance.csv"
imp_df.to_csv(imp_path, index=False)
print("\nSaved feature importance:", imp_path)

# ================= SELECT GOOD FEATURES =================
bad_features = imp_df[imp_df["importance_mean"] < 0]["feature"].tolist()
good_features = [f for f in FEATURES if f not in bad_features]

print("\n================ FEATURE SELECTION =================")
print("Removed BAD features:", bad_features)
print("Remaining GOOD features:", good_features)

# ================= RETRAIN CLEAN MODEL =================
X_train2 = train_df[good_features]
X_test2  = test_df[good_features]

reg2 = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
reg2.fit(X_train2, y_train)

# ================= EVAL CLEAN MODEL =================
clean_pred = test_df["pred_length"] + reg2.predict(X_test2)

clean_mae = mean_absolute_error(test_df["true_length"], clean_pred)
clean_r2  = r2_score(test_df["true_length"], clean_pred)

def pct_within_tol(y_true, y_pred, tol=0.10):
    return np.mean(np.abs(y_true - y_pred) / y_true < tol) * 100

baseline_pct = pct_within_tol(test_df["true_length"], baseline_pred)
clean_pct    = pct_within_tol(test_df["true_length"], clean_pred)

print("\n================ CLEAN MODEL METRICS ================")
print(f"MAE: {clean_mae:.4f}")
print(f"R2 : {clean_r2:.4f}")
print(f"±10% Baseline: {baseline_pct:.2f}")
print(f"±10% Clean   : {clean_pct:.2f}")

# ================= SAVE COMPARISON CSV =================
out_df = test_df[["ptt", "id", "true_length", "pred_length"]].copy()
out_df["baseline_pred"] = baseline_pred
out_df["clean_pred"] = clean_pred

out_df["baseline_correct"] = (np.abs(out_df["true_length"] - out_df["baseline_pred"]) < TOL_MM).astype(int)
out_df["clean_correct"]    = (np.abs(out_df["true_length"] - out_df["clean_pred"]) < TOL_MM).astype(int)

out_df["fixed_by_clean"] = ((out_df["baseline_correct"] == 0) & (out_df["clean_correct"] == 1)).astype(int)
out_df["ruined_by_clean"] = ((out_df["baseline_correct"] == 1) & (out_df["clean_correct"] == 0)).astype(int)

csv_path = OUT_DIR + r"\FEATURE_PRUNING_RESULTS.csv"
out_df.to_csv(csv_path, index=False)

print("\n================ CSV SAVED ================")
print(csv_path)

# ================= SUMMARY =================
print("\n================ FINAL SUMMARY =================")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Clean MAE   : {clean_mae:.4f}")
print(f"Improvement : {(baseline_mae-clean_mae)/baseline_mae*100:.2f}%")
print(f"Fixed cases : {out_df['fixed_by_clean'].sum()}")
print(f"Ruined cases: {out_df['ruined_by_clean'].sum()}")
