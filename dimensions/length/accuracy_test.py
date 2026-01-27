import pandas as pd
import numpy as np

# ================= CONFIG ================= #

# Ground truth CSV (REFERENCE)
REF_FILE  = r"D:\Anubhav\machine_learning_pipelines\resources\12inch_7.1mm.csv"

# Model results CSV (TEST)
TEST_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results\PTT_1_RESULTS.csv"

# Column names in reference CSV
REF_ID_COL   = "S.No"              # <-- IMPORTANT
REF_DIST_COL = "absolute_distance"
REF_ORI_COL  = "orientation"
REF_LEN_COL  = "length"

# Column names in test CSV
TEST_ID_COL   = "id"
TEST_DIST_COL = "absolute_distance"
TEST_ORI_COL  = "orientation"
TEST_LEN_COL  = "length"

# Thresholds
DIST_THRESHOLD_MM = 110
ORI_THRESHOLD_MIN = 80   # minutes
border_threshold_limit = 10

# Unit flags
TEST_DISTANCE_IN_METERS = True
REF_DISTANCE_IN_METERS  = False

# ========================================= #


# ---------- Helpers ---------- #

def orientation_to_minutes(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ["-", "", "nan", "None"]:
        return np.nan
    try:
        h, m = x.split(":")[:2]
        return int(h) * 60 + int(m)
    except:
        return np.nan


def circular_diff(a, b, max_val=720):
    return min(abs(a - b), max_val - abs(a - b))


# ---------- Load CSVs ---------- #

print("\n[INFO] Loading reference CSV...")
ref = pd.read_csv(REF_FILE)
ref = ref[[REF_ID_COL, REF_DIST_COL, REF_ORI_COL, REF_LEN_COL]]

print("[DEBUG] Reference sample:")
print(ref.head())

print("\n[INFO] Loading test CSV...")
test = pd.read_csv(TEST_FILE)
test = test[[TEST_ID_COL, TEST_DIST_COL, TEST_ORI_COL, TEST_LEN_COL]]

print("[DEBUG] Test sample:")
print(test.head())


# ---------- Unit Conversion ---------- #

if TEST_DISTANCE_IN_METERS:
    test["dist_mm"] = test[TEST_DIST_COL] * 1000
else:
    test["dist_mm"] = test[TEST_DIST_COL]

if REF_DISTANCE_IN_METERS:
    ref["dist_mm"] = ref[REF_DIST_COL] * 1000
else:
    ref["dist_mm"] = ref[REF_DIST_COL]


# ---------- Orientation ---------- #

ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)


# ---------- Sort ---------- #

ref  = ref.sort_values("dist_mm").reset_index(drop=True)
test = test.sort_values("dist_mm").reset_index(drop=True)

# Match tracking
ref["matched_flag"] = False
ref["matched_by_test"] = None
ref["matched_dist_diff"] = None
ref["matched_ori_diff"] = None

results = []


# ---------- MATCHING WITH FULL DEBUG ---------- #

print("\n================ MATCHING START =================\n")

for ti, t in test.iterrows():

    print("\n" + "="*120)
    print(f"[TEST] ID={t[TEST_ID_COL]} | dist={t['dist_mm']:.2f} mm | ori={t[TEST_ORI_COL]} | len={t[TEST_LEN_COL]}")
    print("="*120)

    best_ref_idx = None
    best_dist_diff = 1e12
    best_ori_diff = None

    for ri, r in ref.iterrows():

        # If already matched, print who matched it
        if r["matched_flag"]:
            print(f"[REF idx={ri} | S.No={r[REF_ID_COL]}] SKIPPED (already matched by TEST ID={r['matched_by_test']})")
            print(f"        PrevDistDiff={r['matched_dist_diff']} | PrevOriDiff={r['matched_ori_diff']}")
            continue

        # Distance difference
        dist_diff = abs(r["dist_mm"] - t["dist_mm"])
        dist_pass = dist_diff <= DIST_THRESHOLD_MM

        # Orientation difference
        ori_diff = None
        ori_pass = True
        if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
            ori_diff = circular_diff(r["ori_min"], t["ori_min"])
            ori_pass = ori_diff <= ORI_THRESHOLD_MIN + border_threshold_limit

        # Debug print
        print(
            f"[REF idx={ri} | S.No={r[REF_ID_COL]}] ref_dist={r['dist_mm']:.2f} | ref_ori={r[REF_ORI_COL]} | "
            f"dist_diff={dist_diff:.2f} | ori_diff={ori_diff} "
            f"--> DIST_{'OK' if dist_pass else 'FAIL'} & ORI_{'OK' if ori_pass else 'FAIL'}"
        )

        # Filter
        if not dist_pass or not ori_pass:
            continue

        print("        >>> CANDIDATE MATCH")

        # Choose closest distance
        if dist_diff < best_dist_diff:
            best_dist_diff = dist_diff
            best_ref_idx = ri
            best_ori_diff = ori_diff

    # ---------- Final Match ---------- #

    if best_ref_idx is not None:
        ref.loc[best_ref_idx, "matched_flag"] = True
        ref.loc[best_ref_idx, "matched_by_test"] = t[TEST_ID_COL]
        ref.loc[best_ref_idx, "matched_dist_diff"] = best_dist_diff
        ref.loc[best_ref_idx, "matched_ori_diff"] = best_ori_diff

        ref_id = ref.loc[best_ref_idx, REF_ID_COL]
        actual_len = ref.loc[best_ref_idx, REF_LEN_COL]
        matched = "YES"

        print(f"\n>>> FINAL MATCH: TestID={t[TEST_ID_COL]} --> RefIdx={best_ref_idx} (S.No={ref_id})")
        print(f"    RefDist={ref.loc[best_ref_idx,'dist_mm']} | RefOri={ref.loc[best_ref_idx,REF_ORI_COL]} | RefLen={actual_len}")
        print(f"    DistDiff={best_dist_diff} | OriDiff={best_ori_diff}")

    else:
        ref_id = np.nan
        actual_len = np.nan
        matched = "NO"
        print(f"\n>>> NO MATCH: TestID={t[TEST_ID_COL]} marked OUTLIER")

    # Save result row
    results.append({
        "id": t[TEST_ID_COL],
        "ref_id": ref_id,   # <==== NEW COLUMN
        "distance_mm": t["dist_mm"],
        "orientation": t[TEST_ORI_COL],
        "pred_length": t[TEST_LEN_COL],
        "actual_length": actual_len,
        "matched": matched
    })


# ---------- Save Outputs ---------- #

out_df = pd.DataFrame(results)
out_df.to_csv("matching_results_with_ref_id.csv", index=False)

# Save reference audit file
ref.to_csv("reference_match_audit.csv", index=False)

print("\n================ SUMMARY ================")
print("Total test defects:", len(test))
print("Matched:", (out_df["matched"] == "YES").sum())
print("Unmatched:", (out_df["matched"] == "NO").sum())
print("Saved → matching_results_with_ref_id.csv")
print("Saved → reference_match_audit.csv")
