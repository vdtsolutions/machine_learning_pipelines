import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, classification_report

# ================= CONFIG =================
MAX_SENSOR_COUNT = 144
OUTLIER_THRESHOLD_RATIO = 0.6
TOL_MM = 10
N_TRIALS = 5000

# 🔥 MODE SWITCH
RUN_HYPERPARAM_SEARCH = True   # True = search, False = use BEST_PARAMS


# ✅ STORE YOUR BEST PARAMS HERE
BEST_PARAMS = {
    "n_estimators": 236,
    "max_depth": 3,
    "learning_rate": 0.022585190130423278,
    "subsample": 0.7748508755343827,
    "colsample_bytree": 0.7388825917603629,
    "random_state": 42
}


DATA_PATH = r"D:\Anubhav\machine_learning_pipelines\dimensions\length\dataset_generator\MASTER_ML_DATASET_V2.csv"
OUT_DIR = rf"D:\Anubhav\machine_learning_pipelines\dimensions\length\train_test_pipeline\AUTO_CV_n_trials={N_TRIALS}"
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
# OUTLIER DETECTION
# ============================================================
def detect_outliers(df):
    df = df.copy()
    df["outlier_sensor"] = df["sensor_span_ratio"] > OUTLIER_THRESHOLD_RATIO
    df["outlier_length"] = df["true_length"] > 2 * (df["pred_length"] + TOL_MM)

    def label_outlier(r):
        if r.outlier_sensor and r.outlier_length: return "both"
        if r.outlier_sensor: return "sensor"
        if r.outlier_length: return "length"
        return "none"

    df["outlier_type"] = df.apply(label_outlier, axis=1)
    df["outlier"] = ((df.outlier_sensor) | (df.outlier_length)).astype(int)
    return df

df = detect_outliers(df)

# ================= DATA SUMMARY =================
print("\n================ DATA SUMMARY ================")
print(f"Total rows: {len(df)}")
print(f"Total outliers: {df.outlier.sum()}")
print(f"Sensor outliers: {df.outlier_sensor.sum()}")
print(f"Length outliers: {df.outlier_length.sum()}")
print(f"Both-rule outliers: {(df.outlier_type=='both').sum()}")
print("Features used:", FEATURES)

# ================= SAVE OUTLIERS CSV =================
outlier_csv = df[df.outlier==1][["ptt","id","outlier_type"]]
outlier_csv.columns = ["ptt_id","id","reason"]
OUTLIER_PATH = os.path.join(OUT_DIR,"OUTLIERS.csv")
outlier_csv.to_csv(OUTLIER_PATH,index=False)
print("Saved OUTLIERS.csv ->", OUTLIER_PATH)

# ================= TARGET =================
df["target_residual"] = df["true_length"] - df["pred_length"]

# ================= METRIC =================
def pct_within_tol(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) < TOL_MM)

# ================= PTT LIST =================
PTTS = sorted(df["ptt"].unique())

# ================= RANDOM PARAM SEARCH =================
def sample_params():
    return {
        "n_estimators": np.random.randint(200,1200),
        "max_depth": np.random.randint(3,10),
        "learning_rate": np.random.uniform(0.02,0.15),
        "subsample": np.random.uniform(0.7,1.0),
        "colsample_bytree": np.random.uniform(0.7,1.0),
        "random_state": 42
    }



# ============================================================
# HYPERPARAM SEARCH OR SKIP
# ============================================================
trial_log_rows = []
trial_results = []

if RUN_HYPERPARAM_SEARCH:
    print("\n" + "="*80)
    print("🚀 STARTING HYPERPARAMETER SEARCH")
    print("="*80)

    for trial in range(N_TRIALS):
        params = sample_params()

        print("\n" + "="*80)
        print(f"🔥 TRIAL {trial+1}/{N_TRIALS}")
        print("PARAMS:", params)
        print("="*80)

        fold_ml_scores = []
        fold_baseline_scores = []

        for val_ptt in PTTS:
            train_df = df[(df.ptt != val_ptt) & (df.outlier == 0)]
            val_df   = df[(df.ptt == val_ptt) & (df.outlier == 0)]

            if len(val_df) < 10:
                print(f"PTT {val_ptt}: SKIPPED (N={len(val_df)})")
                continue

            X_train = train_df[FEATURES]
            y_train = train_df["target_residual"]
            X_val   = val_df[FEATURES]
            y_val   = val_df["true_length"]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            pred_corrected = val_df["pred_length"] + model.predict(X_val)

            baseline_acc = pct_within_tol(y_val, val_df["pred_length"])
            ml_acc       = pct_within_tol(y_val, pred_corrected)

            fold_baseline_scores.append(baseline_acc)
            fold_ml_scores.append(ml_acc)

            print(f"PTT {val_ptt:>2} | BASELINE={baseline_acc:.4f} | ML={ml_acc:.4f} | N={len(val_df)}")

        mean_baseline = np.mean(fold_baseline_scores)
        mean_ml       = np.mean(fold_ml_scores)

        print("-"*80)
        print(f"TRIAL {trial+1} MEAN BASELINE ACC = {mean_baseline:.6f}")
        print(f"TRIAL {trial+1} MEAN ML ACC       = {mean_ml:.6f}")
        print("-"*80)

        trial_results.append((mean_ml, params))
        trial_log_rows.append({
            "trial_id": trial+1,
            "baseline_cv_accuracy_clean": mean_baseline,
            "ml_cv_accuracy_clean": mean_ml,
            "params": str(params)
        })

    # sort trials
    trial_results.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_params = trial_results[0]

    print("\n" + "="*80)
    print("🏆 BEST TRIAL FOUND")
    print(f"BEST CV ML ACC = {best_acc}")
    print("BEST PARAMS =", best_params)
    print("="*80)

    # save trial log
    trial_log_df = pd.DataFrame(trial_log_rows)
    TRIAL_LOG_PATH = os.path.join(OUT_DIR, "TRIAL_ACCURACY_LOG.csv")
    trial_log_df.to_csv(TRIAL_LOG_PATH, index=False)
    print("Saved TRIAL LOG ->", TRIAL_LOG_PATH)

else:
    print("\n" + "="*80)
    print("⚡ SKIPPING SEARCH — USING FIXED BEST_PARAMS")
    print("="*80)
    best_params = BEST_PARAMS
    print("BEST_PARAMS =", best_params)

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
train_all = df[df.outlier==0]
X_train = train_all[FEATURES]
y_train = train_all["target_residual"]

best_model = XGBRegressor(**best_params)
best_model.fit(X_train,y_train)
print("\nFinal model trained on ALL CLEAN data")

# ============================================================
# APPLY MODEL
# ============================================================
def apply_model(df_in):
    df_in = df_in.copy()
    X = df_in[FEATURES]

    corr = best_model.predict(X)
    corr = np.clip(corr, -0.3*df_in.pred_length, 0.3*df_in.pred_length)

    df_in["pred_corrected"] = df_in.pred_length
    mask = df_in.outlier==0
    df_in.loc[mask,"pred_corrected"] = df_in.loc[mask,"pred_length"] + corr[mask]

    df_in.loc[df_in.pred_length<10,"pred_corrected"] = df_in.pred_length
    return df_in

full_df = apply_model(df)

# ============================================================
# CONSOLIDATED CSV
# ============================================================
full_out = full_df[["ptt","id","true_length","pred_length","pred_corrected","outlier","outlier_type"]].copy()
full_out.columns = ["ptt_id","id","actual_length","pred_length","model_pred_length","outlier","outlier_type"]

full_out.loc[:, "prev_correct"] = (np.abs(full_out.actual_length - full_out.pred_length) < TOL_MM).astype(int)
full_out.loc[:, "after_model_correct"] = (np.abs(full_out.actual_length - full_out.model_pred_length) < TOL_MM).astype(int)

full_out.loc[:, "fixed_by_model"]  = ((full_out.prev_correct == 0) & (full_out.after_model_correct == 1)).astype(int)
full_out.loc[:, "ruined_by_model"] = ((full_out.prev_correct == 1) & (full_out.after_model_correct == 0)).astype(int)


FULL_OUT_PATH = os.path.join(OUT_DIR,"CONSOLIDATED_ALL_PTT_PREDICTIONS.csv")
full_out.to_csv(FULL_OUT_PATH,index=False)
print("Saved CONSOLIDATED CSV ->", FULL_OUT_PATH)

# ============================================================
# PTT ACCURACY CLEAN vs ALL (SAVE CSV)
# ============================================================
ptt_acc_rows = []

print("\n================ PTT ACCURACY CLEAN vs ALL ================")

for ptt, g_all in full_out.groupby("ptt_id"):
    g_clean = g_all[g_all.outlier == 0]

    # CLEAN ACCURACY
    before_clean = g_clean.prev_correct.mean() if len(g_clean) > 0 else np.nan
    after_clean  = g_clean.after_model_correct.mean() if len(g_clean) > 0 else np.nan

    # ALL ACCURACY
    before_all = (np.abs(g_all.actual_length - g_all.pred_length) < TOL_MM).mean()
    after_all  = (np.abs(g_all.actual_length - g_all.model_pred_length) < TOL_MM).mean()

    print(f"PTT {ptt}: CLEAN Before={before_clean:.4f}, After={after_clean:.4f} | "
          f"ALL Before={before_all:.4f}, After={after_all:.4f} | N={len(g_all)}")

    ptt_acc_rows.append({
        "ptt_id": ptt,
        "before_accuracy_clean": before_clean,
        "after_ml_clean": after_clean,
        "before_all": before_all,
        "after_all": after_all,
        "n_samples": len(g_all)
    })

ptt_acc_df = pd.DataFrame(ptt_acc_rows)
PTT_ACC_PATH = os.path.join(OUT_DIR, "PTT_ACCURACY_CLEAN_ALL.csv")
ptt_acc_df.to_csv(PTT_ACC_PATH, index=False)

print("\nSaved PTT_ACCURACY_CLEAN_ALL.csv ->", PTT_ACC_PATH)



# ============================================================
# CONFUSION MATRIX + INTERPRETATION
# ============================================================
clean = full_out[full_out.outlier==0]
y_true = clean.prev_correct
y_pred = clean.after_model_correct

cm = confusion_matrix(y_true,y_pred)

print("\n================ RAW CONFUSION MATRIX (sklearn) ================")
print(cm)

print("""
IMPORTANT INTERPRETATION (CORRECTION SYSTEM):

prev_correct = physics correctness
after_model_correct = ML correctness

Matrix [[TN FP],[FN TP]] means:

TN = STILL WRONG (0 → 0)
FP = FIXED        (0 → 1)  ✅ GOOD
FN = RUINED       (1 → 0)  ❌ BAD
TP = SAFE         (1 → 1)
""")

print("\nClassification report (not super meaningful here):")
print(classification_report(y_true,y_pred,digits=4,zero_division=0))

# ============================================================
# REAL CORRECTION METRICS
# ============================================================
fixed  = ((clean.prev_correct==0)&(clean.after_model_correct==1)).sum()
ruined = ((clean.prev_correct==1)&(clean.after_model_correct==0)).sum()
safe   = ((clean.prev_correct==1)&(clean.after_model_correct==1)).sum()
nofix  = ((clean.prev_correct==0)&(clean.after_model_correct==0)).sum()

print("\n================ REAL CORRECTION METRICS (GLOBAL) ================")
print("FIXED :", fixed)
print("RUINED:", ruined)
print("SAFE  :", safe)
print("NO_FIX:", nofix)

# ============================================================
# PTT SUMMARY
# ============================================================
rows = []
print("\n================ PTT SUMMARY =================")

for ptt,g in clean.groupby("ptt_id"):
    prev_acc = g.prev_correct.mean()
    after_acc = g.after_model_correct.mean()

    fixed  = ((g.prev_correct==0)&(g.after_model_correct==1)).sum()
    ruined = ((g.prev_correct==1)&(g.after_model_correct==0)).sum()
    safe   = ((g.prev_correct==1)&(g.after_model_correct==1)).sum()
    nofix  = ((g.prev_correct==0)&(g.after_model_correct==0)).sum()

    print(f"\nPTT {ptt}")
    print(f"Acc Before={prev_acc:.4f}, After={after_acc:.4f}, N={len(g)}")
    print(f"FIXED={fixed}, RUINED={ruined}, SAFE={safe}, NO_FIX={nofix}")

    rows.append([ptt,prev_acc,after_acc,fixed,ruined,safe,nofix,len(g)])

ptt_df = pd.DataFrame(rows,columns=["ptt_id","baseline_acc","ml_acc","fixed","ruined","safe","nofix","n"])
PTT_PATH = os.path.join(OUT_DIR,"PTT_CORRECTION_SUMMARY.csv")
ptt_df.to_csv(PTT_PATH,index=False)

print("\n================ FILES SAVED ================")
print("OUTLIERS:", OUTLIER_PATH)
print("CONSOLIDATED:", FULL_OUT_PATH)
print("PTT SUMMARY:", PTT_PATH)
