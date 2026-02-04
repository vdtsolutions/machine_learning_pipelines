# ================= RULE FUNCTIONS ================= #
# DEPTH VERSION (width → depth only, logic unchanged)

import pandas as pd


# RULE 1 — UNIQUE ONE-TO-ONE MATCH
def rule_one_single_block(out_df, info, ref_flag):
    flags_updated = 0

    ref_id = info["ref_ids"][0]
    tid = info["test_ids"][0]

    if ref_flag[ref_id] == 0:
        out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = ref_id
        ref_flag[ref_id] = 1
        flags_updated += 1
        print(f"RULE1 ASSIGN: TEST_ID={tid} -> REF_ID={ref_id}")
    else:
        print(f"RULE1 SKIP: REF {ref_id} already used")

    return out_df, ref_flag, flags_updated


# RULE 2 — MANY TESTS → ONE REF (DEPTH ERROR BASED)
def rule_two_single_block(out_df, info, ref_flag):
    flags_updated = 0
    ref_id = info["ref_ids"][0]

    if ref_flag[ref_id] == 1:
        print(f"RULE2 SKIP: REF {ref_id} already used")
        return out_df, ref_flag, 0

    errors = {}
    for tid in info["test_ids"]:
        pred = float(info["pred_depth"][tid])
        actual = float(info["actual_depth"][tid])
        errors[tid] = abs(pred - actual)

    winner = min(errors, key=errors.get)

    out_df.loc[out_df["id"] == winner, "filtered_ids_new"] = ref_id
    ref_flag[ref_id] = 1
    flags_updated += 1

    for tid in info["test_ids"]:
        if tid != winner:
            out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0

    print(f"RULE2 BLOCK REF={ref_id} WINNER TEST_ID={winner} ERRORS={errors}")

    return out_df, ref_flag, flags_updated


# RULE 3 — ONE TEST → MANY REFS (BEST DEPTH MATCH)
def rule_three_single_block(out_df, info, ref_flag):
    flags_updated = 0

    tid = info["test_ids"][0]
    pred = float(info["pred_depth"][tid])
    ref_ids = info["ref_ids"]
    actuals = info["actual_depth"][tid]

    errors = {rid: abs(pred - float(act)) for rid, act in zip(ref_ids, actuals)}
    sorted_refs = sorted(errors, key=errors.get)

    print(f"RULE3 BLOCK TEST_ID={tid} ERRORS={errors}")

    for rid in sorted_refs:
        if ref_flag[rid] == 0:
            out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = rid
            ref_flag[rid] = 1
            flags_updated += 1
            print(f"RULE3 ASSIGN: TEST_ID={tid} -> REF_ID={rid}")
            return out_df, ref_flag, flags_updated

    out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
    print(f"RULE3 FAIL: TEST_ID={tid} no free ref")

    return out_df, ref_flag, 0


# RULE 4 — MANY TESTS ↔ MANY REFS (DISTANCE + DEPTH HEURISTIC)
def rule_four_single_block(out_df, info, ref_flag):
    print("\n[RULE FOUR] Distance-aware multi-test multi-ref resolution")

    flags_updated = 0
    test_ids = info["test_ids"]
    ref_ids = info["ref_ids"]

    # Distances
    distances = {
        tid: out_df.loc[out_df["id"] == tid, "distance_mm"].values[0]
        for tid in test_ids
    }

    unique_distances = set(distances.values())

    # ================= CASE 1: SAME DISTANCE ================= #
    if len(unique_distances) == 1:
        print(f"RULE4 CASE1: SAME DISTANCE for TEST_IDS {test_ids}")

        best_test_error = {}

        for tid in test_ids:
            pred = float(info["pred_depth"][tid])
            actuals = info["actual_depth"][tid]
            best_test_error[tid] = min(abs(pred - float(a)) for a in actuals)

        winner_tid = min(best_test_error, key=best_test_error.get)
        print(f"RULE4 CASE1 WINNER TEST_ID={winner_tid} ERRORS={best_test_error}")

        pred = float(info["pred_depth"][winner_tid])
        actuals = info["actual_depth"][winner_tid]
        ref_errors = {rid: abs(pred - float(act)) for rid, act in zip(ref_ids, actuals)}
        sorted_refs = sorted(ref_errors, key=ref_errors.get)

        for rid in sorted_refs:
            if ref_flag[rid] == 0:
                out_df.loc[out_df["id"] == winner_tid, "filtered_ids_new"] = rid
                ref_flag[rid] = 1
                flags_updated += 1
                print(f"RULE4 CASE1 ASSIGNED REF {rid} TO TEST_ID {winner_tid}")
                break

        for tid in test_ids:
            if tid != winner_tid:
                out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0

        return out_df, ref_flag, flags_updated

    # ================= CASE 2: DIFFERENT DISTANCES ================= #
    print(f"RULE4 CASE2: DIFFERENT DISTANCES {distances}")

    sorted_test_ids = sorted(test_ids, key=lambda t: distances[t])

    for tid in sorted_test_ids:
        pred = float(info["pred_depth"][tid])
        actuals = info["actual_depth"][tid]

        ref_errors = {rid: abs(pred - float(act)) for rid, act in zip(ref_ids, actuals)}
        sorted_refs = sorted(ref_errors, key=ref_errors.get)

        for rid in sorted_refs:
            if ref_flag[rid] == 0:
                out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = rid
                ref_flag[rid] = 1
                flags_updated += 1
                print(f"RULE4 CASE2 ASSIGNED TEST_ID={tid} -> REF={rid}")
                break
        else:
            out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0

    # ================= SANITY CHECK ================= #
    print("\n[RULE4 SANITY CHECK] Distance consistency")

    ref_dist_map = info["ref_dist_map"]
    sorted_test_ids = sorted(test_ids, key=lambda t: distances[t])
    TOL = 0.05

    for i in range(len(sorted_test_ids) - 1):
        t1, t2 = sorted_test_ids[i], sorted_test_ids[i + 1]

        r1 = out_df.loc[out_df["id"] == t1, "filtered_ids_new"].values[0]
        r2 = out_df.loc[out_df["id"] == t2, "filtered_ids_new"].values[0]

        if r1 == 0 or r2 == 0 or pd.isna(r1) or pd.isna(r2):
            continue

        test_gap = abs(distances[t2] - distances[t1])
        ref_gap = abs(ref_dist_map[int(r2)] - ref_dist_map[int(r1)])

        if test_gap < ref_gap * (1 - TOL):
            drop_tid = t2 if int(r1) < int(r2) else t1
            drop_ref = r2 if drop_tid == t2 else r1

            out_df.loc[out_df["id"] == drop_tid, "filtered_ids_new"] = 0
            ref_flag[int(drop_ref)] = 0

    return out_df, ref_flag, flags_updated


# ================= RULE DISPATCHER ================= #
def rule_dispatcher(out_df, block_stats_dict, ref_flag):
    print("\n===== RULE DISPATCHER START =====")
    total_flags = 0

    for block, info in block_stats_dict.items():
        block_size = info["block_size_ref_ids"]
        count_test = info["count_test_ids"]

        if block_size == 1 and count_test == 1:
            out_df, ref_flag, c = rule_one_single_block(out_df, info, ref_flag)
        elif block_size == 1 and count_test > 1:
            out_df, ref_flag, c = rule_two_single_block(out_df, info, ref_flag)
        elif block_size > 1 and count_test == 1:
            out_df, ref_flag, c = rule_three_single_block(out_df, info, ref_flag)
        elif block_size > 1 and count_test > 1:
            out_df, ref_flag, c = rule_four_single_block(out_df, info, ref_flag)
        else:
            c = 0

        total_flags += c

    print(f"\n===== TOTAL FLAGS UPDATED = {total_flags} =====")
    return out_df, ref_flag


# ================= MAIN MAKER FUNCTION ================= #
def filtered_ids_new_maker(out_df, block_stats_dict, max_ref_id):

    ref_flag = {i: 0 for i in range(1, max_ref_id + 1)}
    out_df["filtered_ids_new"] = 0

    out_df, ref_flag = rule_dispatcher(out_df, block_stats_dict, ref_flag)

    used_refs = [r for r, v in ref_flag.items() if v == 1]
    print("\nUSED REF IDS:", used_refs)
    print("TOTAL USED REF COUNT:", len(used_refs))

    return out_df, ref_flag
