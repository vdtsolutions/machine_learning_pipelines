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

PARAMS_TO_EVALUATE = ["length"]

PARAM_COLUMN_MAP = {
    "length": ("length", "length_percent"),
    "depth":  ("Depth",  "depth"),
    "width":  ("Width",  "width_new2")
}

DIST_THRESHOLD_MM = 110
ORI_THRESHOLD_MIN = 80
border_threshold_limit = 10

TEST_DISTANCE_IN_METERS = True
REF_DISTANCE_IN_METERS  = False

# DEBUG CONTROL (None = all, or set like {24})
DEBUG_IDS = None

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

test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")

test["dist_mm"] = test[TEST_DIST_COL] * 1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
ref["dist_mm"]  = ref[REF_DIST_COL]  * 1000 if REF_DISTANCE_IN_METERS  else ref[REF_DIST_COL]

test = test.dropna(subset=["dist_mm"])
test = test[test["dist_mm"] > 0]
ref  = ref.dropna(subset=["dist_mm"])
ref  = ref[ref["dist_mm"] > 0]

ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)

ref  = ref.sort_values("dist_mm").reset_index(drop=True)
test = test.sort_values("dist_mm").reset_index(drop=True)

# ================= MULTI MATCHING ================= #

results = []

print("\n================ MULTI MATCHING DEBUG START ================\n")

for _, t in test.iterrows():
    tid = int(t[TEST_ID_COL])
    dbg = (DEBUG_IDS is None) or (tid in DEBUG_IDS)

    print("\n" + "="*140)
    print(f"[TEST] ID={tid} | dist={t['dist_mm']:.2f} mm | ori={t[TEST_ORI_COL]}")
    print("="*140)

    matched_refs = []

    for ri, r in ref.iterrows():

        dist_diff = abs(r["dist_mm"] - t["dist_mm"])
        dist_pass = dist_diff <= DIST_THRESHOLD_MM

        ori_diff = None
        ori_pass = True
        if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
            ori_diff = circular_diff(r["ori_min"], t["ori_min"])
            ori_pass = ori_diff <= ORI_THRESHOLD_MIN + border_threshold_limit

        print(
            f"[REF idx={ri} | REF_ID={r[REF_ID_COL]}] "
            f"dist_diff={dist_diff:.2f} | ori_diff={ori_diff} "
            f"=> DIST_{'OK' if dist_pass else 'FAIL'} & ORI_{'OK' if ori_pass else 'FAIL'}"
        )

        if not dist_pass or not ori_pass:
            continue

        print("        >>> VALID MATCH")

        matched_refs.append({
            "ref_id": r[REF_ID_COL],
            "dist_diff": dist_diff,
            "ori_diff": ori_diff,
            "ref_length": r["length"]
        })

    print(f"==== TEST {tid} TOTAL MATCHES = {len(matched_refs)} | REFS = {[m['ref_id'] for m in matched_refs]}")

    matched = "YES" if matched_refs else "NO"

    row = {
        "id": tid,
        "ref_ids_all": ",".join(str(m["ref_id"]) for m in matched_refs),
        "ref_lengths_all": ",".join(str(m["ref_length"]) for m in matched_refs),
        "dist_diffs_all": ",".join(str(m["dist_diff"]) for m in matched_refs),
        "ori_diffs_all": ",".join(str(m["ori_diff"]) for m in matched_refs),
        "distance_mm": t["dist_mm"],
        "orientation": t[TEST_ORI_COL],
        "matched": matched,
        "num_matches": len(matched_refs),
        "pred_length": t["length_percent"]
    }

    results.append(row)

out_df = pd.DataFrame(results)

# ================= EXPLODE TABLE ================= #

print("\n================ GLOBAL FILTERING DEBUG ================\n")

for c in ["ref_ids_all","ref_lengths_all","dist_diffs_all","ori_diffs_all"]:
    out_df[c] = out_df[c].astype(str).str.replace(" ", "")

temp = out_df.copy()
for c in ["ref_ids_all","ref_lengths_all","dist_diffs_all","ori_diffs_all"]:
    temp[c] = temp[c].str.split(",")

temp = temp.explode(["ref_ids_all","ref_lengths_all","dist_diffs_all","ori_diffs_all"])
temp.to_csv("PAIRING_TABLE.csv", index=False)
print("Saved PAIRING_TABLE.csv")

# Convert numeric
for c in ["ref_ids_all","ref_lengths_all","dist_diffs_all","ori_diffs_all","pred_length"]:
    temp[c] = pd.to_numeric(temp[c], errors="coerce")

temp["ori_diffs_all"]  = temp["ori_diffs_all"].fillna(0)
temp["dist_diffs_all"] = temp["dist_diffs_all"].fillna(0)
temp = temp.dropna(subset=["ref_ids_all"])

# ================= LENGTH ERROR SCORING ================= #

temp["len_error"] = abs(temp["pred_length"] - temp["ref_lengths_all"])
temp["score"] = temp["len_error"]

temp = temp.sort_values("score")

# Store per-ref len_error table for merge
len_table = temp.groupby("id")["len_error"].apply(lambda x: ",".join(map(str, x))).reset_index()
len_table = len_table.rename(columns={"len_error":"len_error_per_ref"})

# Best len_error per test
best_len = temp.groupby("id")["len_error"].min().reset_index()
best_len = best_len.rename(columns={"len_error":"len_error_best"})

# Merge into out_df
out_df = out_df.merge(len_table, on="id", how="left")
out_df = out_df.merge(best_len, on="id", how="left")

# ================= GREEDY ASSIGNMENT ================= #

assigned_test = set()
assigned_ref  = set()
final_matches = []

# DEBUG AUDIT LOG
assign_log = {}

print("\n=========== ASSIGNMENT TRACE ===========")

for _, row in temp.iterrows():
    t = int(row["id"])
    r = int(row["ref_ids_all"])
    s = row["score"]

    if t in assigned_test:
        print(f"[TEST {t}] SKIP: already assigned")
        continue

    if r in assigned_ref:
        print(f"[TEST {t}] SKIP: REF {r} already taken")
        assign_log.setdefault(t, []).append(f"REF {r} TAKEN")
        continue

    print(f"[TEST {t}] ASSIGN --> REF {r} | len_error={s:.3f}")
    assign_log.setdefault(t, []).append(f"ASSIGNED REF {r}")
    assigned_test.add(t)
    assigned_ref.add(r)
    final_matches.append((t, r, s))

# Write back filtered_ids
out_df["filtered_ids"] = np.nan
out_df["assign_debug"] = ""

for t, r, s in final_matches:
    out_df.loc[out_df["id"] == t, "filtered_ids"] = r

# Fill debug logs
for tid, logs in assign_log.items():
    out_df.loc[out_df["id"]==tid,"assign_debug"] = " | ".join(logs)

# ================= DUPLICATE SUPPRESSION ================= #

out_df["ref_ids_norm"] = out_df["ref_ids_all"].astype(str).apply(
    lambda x: ",".join(sorted(x.split(",")))
)

DIST_TOL = 5

print("\n================ DUPLICATE SUPPRESSION DEBUG ================\n")

for refs, group in out_df.groupby("ref_ids_norm"):

    if len(group) <= 1:
        continue

    group = group.sort_values("distance_mm").copy()
    group["dist_cluster"] = (group["distance_mm"].diff().abs() > DIST_TOL).cumsum()

    for c, cluster in group.groupby("dist_cluster"):
        if len(cluster) <= 1:
            continue

        if cluster["len_error_best"].isna().all():
            best_idx = cluster["distance_mm"].idxmin()
            reason = "DIST_FALLBACK"
        else:
            best_idx = cluster["len_error_best"].idxmin()
            reason = "LEN_ERROR"

        drop_idx = cluster.index.difference([best_idx])

        print(f"[DEDUP] ref_ids={refs} cluster={c} METHOD={reason}")
        print(f"   KEEP TEST_ID={out_df.loc[best_idx,'id']}")
        print(f"   DROP TEST_IDS={list(out_df.loc[drop_idx,'id'])}")

        # MARK WHY DROPPED
        for di in drop_idx:
            tid = out_df.loc[di,"id"]
            out_df.loc[di,"assign_debug"] += f" | DEDUP_REMOVED({reason})"

        out_df.loc[drop_idx, "filtered_ids"] = np.nan


# ================= FINAL TRACE ================= #

print("\n================ FINAL TRACE =================\n")

for _, row in out_df.sort_values("id").iterrows():
    print(
        f"TEST_ID={int(row['id']):03d} | "
        f"matched={row['matched']} | "
        f"ref_ids_all={row['ref_ids_all']} | "
        f"filtered={row['filtered_ids']} | "
        f"pred_len={row['pred_length']} | "
        f"dist={row['distance_mm']} | "
        f"ori={row['orientation']} | "
        f"DEBUG={row['assign_debug']}"
    )

# Safety check
dups = out_df["filtered_ids"].dropna().duplicated().sum()
print("\nDUPLICATE FILTERED IDS =", dups)
assert dups == 0, "🔥 DUPLICATE filtered_ids FOUND"

# Move filtered_ids to LAST column
cols = [c for c in out_df.columns if c != "filtered_ids"] + ["filtered_ids"]
out_df = out_df[cols]

# Save
out_df.to_csv("multi_matching_results_REAL.csv", index=False)

print("\n================ SUMMARY ================")
print("Total test defects:", len(test))
print("Matched:", (out_df["matched"]=="YES").sum())
print("Filtered unique IDs:", out_df["filtered_ids"].notna().sum())
print("Saved → multi_matching_results_REAL.csv")
