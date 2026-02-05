# this script works for both single run and multiple file runs
# just change RUN_MODE between single and batch
# remember the file name should be PTT_1_RESULTS.csv and so on

import re
import os
import glob
import pandas as pd
import numpy as np

# ================= CONFIG ================= #
from defect_matcher.depth_pipeline.filtering_rules_depth import filtered_ids_new_maker

REF_FILE  = r"D:\Anubhav\machine_learning_pipelines\resources\12_inch_5.5mm.csv"

# SINGLE MODE FILE
TEST_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results\PTT_1_RESULTS.csv"

# BATCH MODE FOLDER
TEST_FOLDER = r"D:\Anubhav\testing_bbnew"

RUN_MODE = "batch"   # "single" or "batch"

# ========================================= #

REF_ID_COL   = "S.No"
REF_DIST_COL = "absolute_distance"
REF_ORI_COL  = "orientation"

TEST_ID_COL   = "id"
TEST_DIST_COL = "absolute_distance"
TEST_ORI_COL  = "orientation"

DIST_THRESHOLD_MM = 110
border_distance_limit = 20
ORI_THRESHOLD_MIN = 80
border_threshold_limit = -10

TEST_DISTANCE_IN_METERS = True
REF_DISTANCE_IN_METERS  = False

depth_threshold = 13

# ===== BATCH SUMMARY STORAGE =====
batch_summary = []

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

#add
def circular_diff(a, b, max_val=720):
    return min(abs(a - b), max_val - abs(a - b))


def depth_filtered_cal(ref, out_df):
    # ================= ACTUAL DEPTH FROM FILTERED IDS ================= #

    ref_depth_map = dict(zip(ref["S.No"].astype(int), ref["Depth"]))

    out_df["filtered_ids_new"] = pd.to_numeric(
        out_df["filtered_ids_new"], errors="coerce"
    ).astype("Int64")

    out_df["actual_depth_filtered"] = out_df["filtered_ids_new"].map(ref_depth_map)

    out_df.loc[
        out_df["filtered_ids_new"].isna() | (out_df["filtered_ids_new"] == 0),
        "actual_depth_filtered"
    ] = np.nan

    return out_df


def depth_error_computation(out_df):

    out_df["pred_depth"] = pd.to_numeric(out_df["pred_depth"], errors="coerce")

    out_df["actual_depth"] = out_df["actual_depth_filtered"]

    out_df["depth_diff"] = out_df["actual_depth_filtered"] - out_df["pred_depth"]

    return out_df


def correct_incorrect_mapping(threshold, out_df):
    THRESHOLD = threshold

    out_df["correct"] = np.where(
        out_df["depth_diff"].abs() <= THRESHOLD,
        "yes",
        "no"
    )

    out_df.loc[out_df["depth_diff"].isna(), "correct"] = np.nan

    return out_df


# ================= MAIN PROCESS FUNCTION ================= #

def process_test_file(TEST_FILE):

    print(f"\n================ RUNNING FILE: {TEST_FILE} =================")

    ptt_name = re.search(r"(PTT_\d+)", TEST_FILE).group(1)
    print("PTT NAME:", ptt_name)

    # ---------- Load ---------- #
    print("[INFO] Loading CSVs...")
    ref  = pd.read_csv(REF_FILE)
    test = pd.read_csv(TEST_FILE)

    # Distance conversion
    test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
    ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")

    test["dist_mm"] = test[TEST_DIST_COL] * 1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
    ref["dist_mm"]  = ref[REF_DIST_COL]  * 1000 if REF_DISTANCE_IN_METERS  else ref[REF_DIST_COL]

    ref_dist_map = dict(zip(ref["S.No"].astype(int), ref["dist_mm"]))

    test = test.dropna(subset=["dist_mm"])
    ref  = ref.dropna(subset=["dist_mm"])

    # Orientation
    ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
    test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)

    ref  = ref.sort_values("dist_mm").reset_index(drop=True)
    test = test.sort_values("dist_mm").reset_index(drop=True)

    # ================= MULTI MATCHING ================= #

    results = []

    for _, t in test.iterrows():
        tid = int(t[TEST_ID_COL])
        matched_refs = []

        for _, r in ref.iterrows():
            dist_diff = abs(r["dist_mm"] - t["dist_mm"])
            dist_pass = dist_diff <= DIST_THRESHOLD_MM

            ori_pass = True
            if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
                ori_diff = circular_diff(r["ori_min"], t["ori_min"])
                ori_pass = ori_diff <= ORI_THRESHOLD_MIN + border_threshold_limit

            if not dist_pass or not ori_pass:
                continue

            matched_refs.append({
                "ref_id": r[REF_ID_COL],
                "ref_depth": r["Depth"]
            })

        row = {
            "id": tid,
            "ref_ids_all": ",".join(str(m["ref_id"]) for m in matched_refs),
            "ref_depths_all": ",".join(str(m["ref_depth"]) for m in matched_refs),
            "distance_mm": t["dist_mm"],
            "orientation": t[TEST_ORI_COL],
            "pred_depth": t["depth_new"]
        }

        results.append(row)

    out_df = pd.DataFrame(results)

    out_df["ref_ids_norm"] = out_df["ref_ids_all"].astype(str).apply(
        lambda x: ",".join(sorted(x.split(","))) if x != "" else ""
    )

    # ================= BUILD BLOCK STATS ================= #

    df_clean = out_df[out_df["ref_ids_norm"] != ""].copy()
    block_id_map = df_clean.groupby("ref_ids_norm")["id"].apply(list).to_dict()

    def sort_key(x):
        return int(x.split(",")[0])

    ref_depth_map = dict(zip(ref["S.No"], ref["Depth"]))
    final_block_stats = {}

    for block in sorted(block_id_map.keys(), key=sort_key):
        test_ids = block_id_map[block]
        ref_list = [int(x) for x in block.split(",")]

        pred_depth_map = {
            tid: out_df.loc[out_df["id"] == tid, "pred_depth"].values[0]
            for tid in test_ids
        }

        actual_depth_map = {}
        for tid in test_ids:
            if len(ref_list) == 1:
                actual_depth_map[tid] = ref_depth_map.get(ref_list[0], np.nan)
            else:
                actual_depth_map[tid] = [ref_depth_map.get(r, np.nan) for r in ref_list]

        final_block_stats[block] = {
            "count_test_ids": len(test_ids),
            "block_size_ref_ids": len(ref_list),
            "ref_ids": ref_list,
            "test_ids": test_ids,
            "pred_depth": pred_depth_map,
            "actual_depth": actual_depth_map,
            "ref_dist_map": ref_dist_map
        }

    max_ref_id = ref["S.No"].max()
    out_df["filtered_ids_new"] = 0

    out_df, ref_flag = filtered_ids_new_maker(out_df, final_block_stats, max_ref_id)

    out_df = depth_filtered_cal(ref, out_df)
    out_df = depth_error_computation(out_df)
    out_df = correct_incorrect_mapping(depth_threshold, out_df)

    # ================= SUMMARY ================= #

    valid_df = out_df[~out_df["depth_diff"].isna()]

    total_correct = (valid_df["correct"] == "yes").sum()
    total_incorrect = (valid_df["correct"] == "no").sum()
    total = len(valid_df)

    accuracy = total_correct / total if total > 0 else np.nan

    if RUN_MODE == "batch":
        batch_summary.append({
            "PTT_NAME": ptt_name,
            "CORRECT": total_correct,
            "INCORRECT": total_incorrect,
            "ACCURACY": accuracy
        })

    # ================= SAVE ================= #

    keep_cols = [
        "id",
        "distance_mm",
        "orientation",
        "ref_ids_all",
        "ref_depths_all",
        "filtered_ids_new",
        "actual_depth_filtered",
        "pred_depth",
        "depth_diff",
        "correct",
        "summary_value",
    ]

    save_dir = os.path.join(os.getcwd(), "defect_match_depth_data")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"results_matching_depth_{ptt_name}.csv")

    summary_rows = pd.DataFrame({
        "correct": ["total_correct", "total_incorrect", "accuracy"],
        "summary_value": [total_correct, total_incorrect, accuracy]
    })

    blank_rows = pd.DataFrame([{}] * 3)

    out_df_with_summary = pd.concat([out_df, blank_rows, summary_rows], ignore_index=True)
    out_df_with_summary[keep_cols].to_csv(save_path, index=False)

    print(f"\n[SAVED] {save_path}")


# ================= RUNNER ================= #

if __name__ == "__main__":

    if RUN_MODE == "single":
        process_test_file(TEST_FILE)

    elif RUN_MODE == "batch":
        all_files = glob.glob(os.path.join(TEST_FOLDER, "PTT_*_RESULTS.csv"))

        for f in sorted(all_files):
            try:
                process_test_file(f)
            except Exception as e:
                print(f"[ERROR] {f} FAILED → {e}")

        summary_df = pd.DataFrame(batch_summary)

        summary_save_path = os.path.join(
            os.getcwd(),
            "defect_match_depth_data",
            f"PTT_SUMMARY_depth_threshold={depth_threshold}.csv"
        )
        summary_df.to_csv(summary_save_path, index=False)

        print("\n[BATCH SUMMARY SAVED] →", summary_save_path)
