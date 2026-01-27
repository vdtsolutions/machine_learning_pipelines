# import pandas as pd
# import numpy as np
#
# # ================= CONFIG ================= #
#
# REF_FILE  = r"D:\Anubhav\machine_learning_pipelines\resources\12inch_7.1mm.csv"
# TEST_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results\PTT_1_RESULTS.csv"
#
# REF_ID_COL   = "S.No"
# REF_DIST_COL = "absolute_distance"
# REF_ORI_COL  = "orientation"
# REF_LEN_COL  = "length"
#
# TEST_ID_COL   = "id"
# TEST_DIST_COL = "absolute_distance"
# TEST_ORI_COL  = "orientation"
# TEST_LEN_COL  = "length"
#
# DIST_THRESHOLD_MM = 110
# ORI_THRESHOLD_MIN = 80
# border_threshold_limit = 10
#
# TEST_DISTANCE_IN_METERS = True
# REF_DISTANCE_IN_METERS  = False
#
# # ========================================= #
#
#
# # ---------- Helpers ---------- #
#
# def orientation_to_minutes(x):
#     if pd.isna(x):
#         return np.nan
#     x = str(x).strip()
#     if x in ["-", "", "nan", "None"]:
#         return np.nan
#     try:
#         h, m = x.split(":")[:2]
#         return int(h) * 60 + int(m)
#     except:
#         return np.nan
#
#
# def circular_diff(a, b, max_val=720):
#     return min(abs(a - b), max_val - abs(a - b))
#
#
# # ---------- Load CSVs ---------- #
#
# print("\n[INFO] Loading reference CSV...")
# ref = pd.read_csv(REF_FILE)[[REF_ID_COL, REF_DIST_COL, REF_ORI_COL, REF_LEN_COL]]
#
# print("\n[INFO] Loading test CSV...")
# test = pd.read_csv(TEST_FILE)[[TEST_ID_COL, TEST_DIST_COL, TEST_ORI_COL, TEST_LEN_COL]]
#
#
# # ---------- NaN & Type Fix (ONLY ADDITION) ---------- #
#
# # Force distance to numeric (convert garbage strings to NaN)
# test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
# ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")
#
#
# # ---------- Unit Conversion ---------- #
#
# test["dist_mm"] = test[TEST_DIST_COL] * 1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
# ref["dist_mm"]  = ref[REF_DIST_COL]  * 1000 if REF_DISTANCE_IN_METERS  else ref[REF_DIST_COL]
#
#
# # ---------- Drop corrupt rows (ONLY ADDITION) ---------- #
#
# test = test.dropna(subset=["dist_mm"])
# test = test[test["dist_mm"] > 0]
#
# ref  = ref.dropna(subset=["dist_mm"])
# ref  = ref[ref["dist_mm"] > 0]
#
#
# # ---------- Orientation ---------- #
#
# ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
# test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)
#
#
# # ---------- Sort ---------- #
#
# ref  = ref.sort_values("dist_mm").reset_index(drop=True)
# test = test.sort_values("dist_mm").reset_index(drop=True)
#
#
# # ---------- Matching ---------- #
#
# results = []
#
# print("\n================ MULTI MATCHING START =================\n")
#
# for _, t in test.iterrows():
#
#     print("\n" + "="*120)
#     print(f"[TEST] ID={t[TEST_ID_COL]} | dist={t['dist_mm']:.2f} mm | ori={t[TEST_ORI_COL]} | len={t[TEST_LEN_COL]}")
#     print("="*120)
#
#     matched_refs = []   # store ALL matching refs
#
#     for ri, r in ref.iterrows():
#
#         # Distance check
#         dist_diff = abs(r["dist_mm"] - t["dist_mm"])
#         dist_pass = dist_diff <= DIST_THRESHOLD_MM
#
#         # Orientation check
#         ori_diff = None
#         ori_pass = True
#
#         if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
#             ori_diff = circular_diff(r["ori_min"], t["ori_min"])
#             ori_pass = ori_diff <= ORI_THRESHOLD_MIN + border_threshold_limit
#
#         print(
#             f"[REF idx={ri} | S.No={r[REF_ID_COL]}] ref_dist={r['dist_mm']:.2f} | ref_ori={r[REF_ORI_COL]} | "
#             f"dist_diff={dist_diff:.2f} | ori_diff={ori_diff} "
#             f"--> DIST_{'OK' if dist_pass else 'FAIL'} & ORI_{'OK' if ori_pass else 'FAIL'}"
#         )
#
#         if not dist_pass or not ori_pass:
#             continue
#
#         print("        >>> VALID MATCH")
#         matched_refs.append({
#             "ref_id": r[REF_ID_COL],
#             "dist_diff": dist_diff,
#             "ori_diff": ori_diff,
#             "actual_len": r[REF_LEN_COL]
#         })
#
#     # ---------- Store Results ---------- #
#
#     if matched_refs:
#         matched = "YES"
#         ref_ids = [m["ref_id"] for m in matched_refs]
#         best_match = min(matched_refs, key=lambda x: x["dist_diff"])  # closest one
#
#         print(f"\n>>> MATCHED REF IDs: {ref_ids}")
#         print(f"    BEST REF = {best_match['ref_id']} | DistDiff={best_match['dist_diff']} | OriDiff={best_match['ori_diff']}")
#
#     else:
#         matched = "NO"
#         ref_ids = []
#         best_match = {"actual_len": np.nan}
#
#         print("\n>>> NO MATCH FOUND")
#
#     results.append({
#         "id": t[TEST_ID_COL],
#         "ref_ids_all": ",".join(map(str, ref_ids)),   # all matches
#         "best_ref_id": best_match.get("ref_id", np.nan),
#         "distance_mm": t["dist_mm"],
#         "orientation": t[TEST_ORI_COL],
#         "pred_length": t[TEST_LEN_COL],
#         "actual_length_best": best_match["actual_len"],
#         "matched": matched,
#         "num_matches": len(ref_ids)
#     })
#
#
# # ---------- Save Outputs ---------- #
#
# out_df = pd.DataFrame(results)
# out_df.to_csv("multi_matching_results.csv", index=False)
#
# print("\n================ SUMMARY ================")
# print("Total test defects:", len(test))
# print("Matched:", (out_df["matched"] == "YES").sum())
# print("Unmatched:", (out_df["matched"] == "NO").sum())
# print("Saved → multi_matching_results.csv")

import pandas as pd
import numpy as np

# ================= CONFIG ================= #

REF_FILE  = r"D:\Anubhav\machine_learning_pipelines\resources\12inch_7.1mm.csv"
TEST_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results\PTT_1_RESULTS.csv"

REF_ID_COL   = "S.No"
REF_DIST_COL = "absolute_distance"
REF_ORI_COL  = "orientation"

TEST_ID_COL   = "id"
TEST_DIST_COL = "absolute_distance"
TEST_ORI_COL  = "orientation"

# 🔥 PARAMETERS TO EVALUATE (CHANGE THIS ONLY)
PARAMS_TO_EVALUATE = ["length"]
# Example: ["length"], ["depth"], ["width"], ["length","depth"]

# Column mapping: param_name -> (REF_COL, TEST_COL)
PARAM_COLUMN_MAP = {
    "length": ("length", "length"),
    "depth":  ("Depth",  "depth"),
    "width":  ("Width",  "width_new2")
}

DIST_THRESHOLD_MM = 110
ORI_THRESHOLD_MIN = 80
border_threshold_limit = 10

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

print("\n[INFO] Loading test CSV...")
test = pd.read_csv(TEST_FILE)


# ---------- Force numeric distance ---------- #

test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")


# ---------- Unit Conversion ---------- #

test["dist_mm"] = test[TEST_DIST_COL] * 1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
ref["dist_mm"]  = ref[REF_DIST_COL]  * 1000 if REF_DISTANCE_IN_METERS  else ref[REF_DIST_COL]


# ---------- Drop corrupt rows ---------- #

test = test.dropna(subset=["dist_mm"])
test = test[test["dist_mm"] > 0]

ref  = ref.dropna(subset=["dist_mm"])
ref  = ref[ref["dist_mm"] > 0]


# ---------- Orientation ---------- #

ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)


# ---------- Sort ---------- #

ref  = ref.sort_values("dist_mm").reset_index(drop=True)
test = test.sort_values("dist_mm").reset_index(drop=True)


# ---------- MATCHING ---------- #

results = []

print("\n================ MULTI MATCHING START =================\n")

for _, t in test.iterrows():

    print("\n" + "="*120)
    print(f"[TEST] ID={t[TEST_ID_COL]} | dist={t['dist_mm']:.2f} mm | ori={t[TEST_ORI_COL]}")
    print("="*120)

    matched_refs = []

    for ri, r in ref.iterrows():

        # Distance check
        dist_diff = abs(r["dist_mm"] - t["dist_mm"])
        dist_pass = dist_diff <= DIST_THRESHOLD_MM

        # Orientation check
        ori_diff = None
        ori_pass = True
        if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
            ori_diff = circular_diff(r["ori_min"], t["ori_min"])
            ori_pass = ori_diff <= ORI_THRESHOLD_MIN + border_threshold_limit

        print(
            f"[REF idx={ri} | S.No={r[REF_ID_COL]}] ref_dist={r['dist_mm']:.2f} | ref_ori={r[REF_ORI_COL]} | "
            f"dist_diff={dist_diff:.2f} | ori_diff={ori_diff} "
            f"--> DIST_{'OK' if dist_pass else 'FAIL'} & ORI_{'OK' if ori_pass else 'FAIL'}"
        )

        if not dist_pass or not ori_pass:
            continue

        print("        >>> VALID MATCH")

        # Collect dynamic actual parameters
        actual_params = {}
        for p in PARAMS_TO_EVALUATE:
            ref_col, _ = PARAM_COLUMN_MAP[p]
            actual_params[p] = r.get(ref_col, np.nan)

        matched_refs.append({
            "ref_id": r[REF_ID_COL],
            "dist_diff": dist_diff,
            "ori_diff": ori_diff,
            "actual_params": actual_params
        })


    # ---------- Select Best Match ---------- #

    if matched_refs:
        matched = "YES"
        ref_ids = [m["ref_id"] for m in matched_refs]
        best_match = min(matched_refs, key=lambda x: x["dist_diff"])
    else:
        matched = "NO"
        ref_ids = []
        best_match = {"ref_id": np.nan, "actual_params": {}}

        print("\n>>> NO MATCH FOUND")


    # ---------- Store Results ---------- #

    result_row = {
        "id": t[TEST_ID_COL],
        "ref_ids_all": ",".join(map(str, ref_ids)),
        "best_ref_id": best_match.get("ref_id", np.nan),
        "distance_mm": t["dist_mm"],
        "orientation": t[TEST_ORI_COL],
        "matched": matched,
        "num_matches": len(ref_ids)
    }

    # Add predicted & actual params dynamically
    for p in PARAMS_TO_EVALUATE:
        _, test_col = PARAM_COLUMN_MAP[p]
        result_row[f"pred_{p}"]   = t.get(test_col, np.nan)
        result_row[f"actual_{p}"] = best_match.get("actual_params", {}).get(p, np.nan)

    results.append(result_row)


# ---------- Save Outputs ---------- #

out_df = pd.DataFrame(results)
out_df.to_csv("multi_matching_results.csv", index=False)

print("\n================ SUMMARY ================")
print("Total test defects:", len(test))
print("Matched:", (out_df["matched"] == "YES").sum())
print("Unmatched:", (out_df["matched"] == "NO").sum())
print("Saved → multi_matching_results.csv")
