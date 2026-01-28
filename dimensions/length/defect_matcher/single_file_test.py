import re

import pandas as pd
import numpy as np


# ================= CONFIG ================= #
from dimensions.length.defect_matcher.filtering_rules import filtered_ids_new_maker

REF_FILE  = r"D:\Anubhav\machine_learning_pipelines\resources\12inch_7.1mm.csv"
TEST_FILE = r"D:\Anubhav\machine_learning_pipelines\resources\results\12\bbnew_results\PTT_2_RESULTS.csv"
ptt_name = re.search(r"(PTT_\d+)", TEST_FILE).group(1)
print(ptt_name)  # PTT_2


REF_ID_COL   = "S.No"
REF_DIST_COL = "absolute_distance"
REF_ORI_COL  = "orientation"

TEST_ID_COL   = "id"
TEST_DIST_COL = "absolute_distance"
TEST_ORI_COL  = "orientation"

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


# ---------- Load ---------- #

print("[INFO] Loading CSVs...")
ref  = pd.read_csv(REF_FILE)
test = pd.read_csv(TEST_FILE)

# Convert distances
test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")

test["dist_mm"] = test[TEST_DIST_COL] * 1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
ref["dist_mm"]  = ref[REF_DIST_COL]  * 1000 if REF_DISTANCE_IN_METERS  else ref[REF_DIST_COL]

# Clean
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
            "ref_length": r["length"]
        })

    row = {
        "id": tid,
        "ref_ids_all": ",".join(str(m["ref_id"]) for m in matched_refs),
        "ref_lengths_all": ",".join(str(m["ref_length"]) for m in matched_refs),
        "distance_mm": t["dist_mm"],
        "orientation": t[TEST_ORI_COL],
        "pred_length": t["length_percent"]
    }

    results.append(row)

out_df = pd.DataFrame(results)

# Normalize block key
out_df["ref_ids_norm"] = out_df["ref_ids_all"].astype(str).apply(
    lambda x: ",".join(sorted(x.split(","))) if x != "" else ""
)

# ================= BUILD BLOCK STATS ================= #

df_clean = out_df[out_df["ref_ids_norm"] != ""].copy()
block_id_map = df_clean.groupby("ref_ids_norm")["id"].apply(list).to_dict()

def sort_key(x):
    return int(x.split(",")[0])

ref_len_map = dict(zip(ref["S.No"], ref["length"]))

final_block_stats = {}

for block in sorted(block_id_map.keys(), key=sort_key):
    test_ids = block_id_map[block]
    ref_list = [int(x) for x in block.split(",")]

    pred_len_map = {tid: out_df.loc[out_df["id"] == tid, "pred_length"].values[0] for tid in test_ids}

    actual_len_map = {}
    for tid in test_ids:
        if len(ref_list) == 1:
            actual_len_map[tid] = ref_len_map.get(ref_list[0], np.nan)
        else:
            actual_len_map[tid] = [ref_len_map.get(r, np.nan) for r in ref_list]

    final_block_stats[block] = {
        "count_test_ids": len(test_ids),
        "block_size_ref_ids": len(ref_list),
        "ref_ids": ref_list,
        "test_ids": test_ids,
        "pred_length": pred_len_map,
        "actual_length": actual_len_map
    }

# ================= APPLY RULE ENGINE ================= #

max_ref_id = ref["S.No"].max()
out_df["filtered_ids_new"] = 0

out_df, ref_flag = filtered_ids_new_maker(out_df, final_block_stats, max_ref_id)


# ================= SAVE CLEAN OUTPUT ================= #

keep_cols = [
    "id",
    "distance_mm",
    "orientation",
    "pred_length",
    "ref_ids_all",
    "ref_lengths_all",
    "filtered_ids_new"
]
import os

# folder in the directory where you run the script from
save_dir = os.path.join(os.getcwd(), "defect_match")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"results_matching_{ptt_name}.csv")
out_df[keep_cols].to_csv(save_path, index=False)

print("\nSaved → filtered_output_clean.csv")
print("Total test defects:", len(out_df))
print("Assigned:", (out_df["filtered_ids_new"] != 0).sum())
