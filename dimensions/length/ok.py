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

DIST_THRESHOLD_MM = 110
ORI_THRESHOLD_MIN = 80
border_threshold_limit = 10

TEST_DISTANCE_IN_METERS = True
REF_DISTANCE_IN_METERS  = False

DIST_PHYS_TOL = 2   # mm physical duplicate tolerance

# ========================================= #

def orientation_to_minutes(x):
    if pd.isna(x): return np.nan
    x = str(x).strip()
    if x in ["-", "", "nan", "None"]: return np.nan
    try:
        h, m = x.split(":")[:2]
        return int(h)*60 + int(m)
    except:
        return np.nan

def circular_diff(a, b, max_val=720):
    return min(abs(a-b), max_val-abs(a-b))

# ---------- Load ---------- #

ref  = pd.read_csv(REF_FILE)
test = pd.read_csv(TEST_FILE)

test[TEST_DIST_COL] = pd.to_numeric(test[TEST_DIST_COL], errors="coerce")
ref[REF_DIST_COL]   = pd.to_numeric(ref[REF_DIST_COL], errors="coerce")

test["dist_mm"] = test[TEST_DIST_COL]*1000 if TEST_DISTANCE_IN_METERS else test[TEST_DIST_COL]
ref["dist_mm"]  = ref[REF_DIST_COL]*1000 if REF_DISTANCE_IN_METERS else ref[REF_DIST_COL]

test = test.dropna(subset=["dist_mm"])
ref  = ref.dropna(subset=["dist_mm"])

test = test[test["dist_mm"] > 0]
ref  = ref[ref["dist_mm"] > 0]

ref["ori_min"]  = ref[REF_ORI_COL].apply(orientation_to_minutes)
test["ori_min"] = test[TEST_ORI_COL].apply(orientation_to_minutes)

ref  = ref.sort_values("dist_mm").reset_index(drop=True)
test = test.sort_values("dist_mm").reset_index(drop=True)

# ================= FLAGS ================= #

MAX_REF_ID = int(ref[REF_ID_COL].max())
greedy_flag = np.zeros(MAX_REF_ID+1, dtype=bool)
final_flag  = np.zeros(MAX_REF_ID+1, dtype=bool)

# ================= BUILD MATCH TABLE ================= #

rows = []

for _, t in test.iterrows():
    tid = int(t[TEST_ID_COL])
    matches = []

    for _, r in ref.iterrows():
        dist_ok = abs(r["dist_mm"] - t["dist_mm"]) <= DIST_THRESHOLD_MM
        ori_ok = True

        if not np.isnan(r["ori_min"]) and not np.isnan(t["ori_min"]):
            ori_ok = circular_diff(r["ori_min"], t["ori_min"]) <= ORI_THRESHOLD_MIN + border_threshold_limit

        if dist_ok and ori_ok:
            matches.append((int(r[REF_ID_COL]), float(r["length"])))

    rows.append({
        "id": tid,
        "distance_mm": t["dist_mm"],
        "pred_length": float(t["length_percent"]),
        "matches": matches
    })

out_df = pd.DataFrame(rows)

# ================= EXPLODE ================= #

temp = []
for _, r in out_df.iterrows():
    for rid, rlen in r["matches"]:
        temp.append((r["id"], rid, rlen, r["pred_length"]))
temp = pd.DataFrame(temp, columns=["id","ref_id","ref_length","pred_length"])
temp["len_error"] = abs(temp["pred_length"] - temp["ref_length"])
temp = temp.sort_values("len_error")

# ================= GREEDY ASSIGN ================= #

assigned_test = set()
out_df["filtered_ids"] = np.nan
out_df["assigned_ref_length"] = np.nan
out_df["debug"] = ""

for _, row in temp.iterrows():
    t = int(row["id"])
    r = int(row["ref_id"])
    rl = row["ref_length"]

    if t in assigned_test:
        continue
    if greedy_flag[r]:
        continue

    greedy_flag[r] = True
    assigned_test.add(t)

    out_df.loc[out_df["id"]==t,"filtered_ids"] = r
    out_df.loc[out_df["id"]==t,"assigned_ref_length"] = rl
    out_df.loc[out_df["id"]==t,"debug"] = f"GREEDY->{r}"

# ================= PHYSICAL DEDUP ================= #

phys_dropped = set()

out_df = out_df.sort_values("distance_mm").reset_index(drop=True)
out_df["phys_cluster"] = (out_df["distance_mm"].diff().abs() > DIST_PHYS_TOL).cumsum()

for c, cluster in out_df.groupby("phys_cluster"):

    cluster = cluster[cluster["filtered_ids"].notna()]
    if len(cluster) <= 1:
        continue

    cluster = cluster.copy()
    cluster["len_error"] = abs(cluster["pred_length"] - cluster["assigned_ref_length"])
    keep_idx = cluster["len_error"].idxmin()
    drop_idx = cluster.index.difference([keep_idx])

    for di in drop_idx:
        tid = int(out_df.loc[di,"id"])
        rid = int(out_df.loc[di,"filtered_ids"])

        phys_dropped.add(tid)
        greedy_flag[rid] = False
        final_flag[rid] = False

        out_df.loc[di,"filtered_ids"] = np.nan
        out_df.loc[di,"debug"] += " | PHYS_DROP"

# ================= REBUILD FINAL FLAG ================= #

final_flag[:] = False
for r in out_df["filtered_ids"].dropna().astype(int):
    final_flag[r] = True

# ================= SECOND PASS REASSIGN ================= #

for _, row in out_df[out_df["filtered_ids"].isna()].iterrows():
    t = int(row["id"])
    if t in phys_dropped:
        continue

    cand = temp[temp["id"]==t]
    for _, c in cand.iterrows():
        r = int(c["ref_id"])
        rl = c["ref_length"]

        if final_flag[r]:
            continue

        out_df.loc[out_df["id"]==t,"filtered_ids"] = r
        out_df.loc[out_df["id"]==t,"assigned_ref_length"] = rl
        out_df.loc[out_df["id"]==t,"debug"] += f" | REASSIGN->{r}"
        final_flag[r] = True
        break

# ================= FINAL CHECK ================= #

dups = out_df["filtered_ids"].dropna().duplicated().sum()
print("DUPLICATE REFS =", dups)
assert dups == 0

# ================= SAVE ================= #

out_df.to_csv("FINAL_DEFECT_MATCHING.csv", index=False)
print("SAVED: FINAL_DEFECT_MATCHING.csv")
