

# def filtered_ids_new_maker(out_df, block_stats_dict, max_ref_id):
#     # Initialize reference flags
#     ref_flag = {i: 0 for i in range(1, max_ref_id + 1)}
#     print(f"\n[INIT] reference flags initialized : {ref_flag}")
#
#     print("\n[INIT] block stats dict:")
#     print(block_stats_dict)
#
#     # Initialize filtered_ids_new = 0 for all involved test_ids
#     for block in block_stats_dict:
#         for tid in block_stats_dict[block]["test_ids"]:
#             out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
#
#     # CALL RULE ONE
#     out_df, ref_flag, rule1_flag_count = rule_one(out_df, block_stats_dict, ref_flag)
#     print(f"\nTOTAL FLAGS UPDATED IN RULE ONE = {rule1_flag_count}")
#     print(f"flags values after running rule one when block size of refs_ids is 1 and count of test_ids is 1: {ref_flag}")
#     out_df, ref_flag, rule2_flag_count = rule_two(out_df, block_stats_dict, ref_flag)
#     print(f"flags values after running rule one when block size of refs_ids is 1 and count of test_ids > 1: {ref_flag}")
#     print(f"\nTOTAL FLAGS UPDATED IN RULE TWO = {rule2_flag_count}")
#     out_df, ref_flag, rule3_flag_count = rule_three(out_df, block_stats_dict, ref_flag)
#     print(f"flags values after running rule one when block size of refs_ids > 1 and count of test_ids = 1: {ref_flag}")
#     print(f"\nTOTAL FLAGS UPDATED IN RULE THREE = {rule3_flag_count}")
#     return out_df, ref_flag
#
#
# def rule_one(out_df, block_stats_dict, ref_flag):
#     print("\n[RULE ONE] Assigning unique blocks (block_size=1 & count=1)")
#
#     flags_updated = 0   # <-- COUNTER
#
#     for block, info in block_stats_dict.items():
#         block_size = info["block_size_ref_ids"]
#         count_test = info["count_test_ids"]
#
#         # Only rule-one blocks
#         if block_size == 1 and count_test == 1:
#
#             ref_id = info["ref_ids"][0]     # single ref
#             tid = info["test_ids"][0]       # single test
#
#             # Safety check: ref must be unused
#             if ref_flag[ref_id] == 0:
#                 out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = ref_id
#                 ref_flag[ref_id] = 1
#                 flags_updated += 1   # <-- COUNT FLAG UPDATE
#
#                 print(f"RULE1 ASSIGN: TEST_ID={tid} -> REF_ID={ref_id}")
#             else:
#                 print(f"RULE1 SKIP: REF_ID={ref_id} already used for TEST_ID={tid}")
#
#     print(f"\n[RULE ONE] FLAGS UPDATED = {flags_updated}")
#     return out_df, ref_flag, flags_updated
#
# def rule_two(out_df, block_stats_dict, ref_flag):
#     print("\n[RULE TWO] Resolving collisions for block_size=1 & count>1")
#
#     flags_updated = 0   # <-- COUNTER
#
#     for block, info in block_stats_dict.items():
#         block_size = info["block_size_ref_ids"]
#         count_test = info["count_test_ids"]
#
#         # Rule Two condition
#         if block_size == 1 and count_test > 1:
#             ref_id = info["ref_ids"][0]
#
#             # Skip if ref already used
#             if ref_flag[ref_id] == 1:
#                 print(f"RULE2 SKIP: REF {ref_id} already used")
#                 continue
#
#             # Compute errors
#             errors = {}
#             for tid in info["test_ids"]:
#                 pred = float(info["pred_length"][tid])
#                 actual = float(info["actual_length"][tid])
#                 errors[tid] = abs(pred - actual)
#
#             # Winner
#             winner_tid = min(errors, key=errors.get)
#
#             print(f"\nRULE2 BLOCK {block} REF {ref_id}")
#             print(f"    ERRORS: {errors}")
#             print(f"    WINNER: TEST_ID {winner_tid}")
#
#             # Assign winner
#             out_df.loc[out_df["id"] == winner_tid, "filtered_ids_new"] = ref_id
#             ref_flag[ref_id] = 1
#             flags_updated += 1   # <-- COUNT FLAG UPDATE
#
#             # Reset losers
#             for tid in info["test_ids"]:
#                 if tid != winner_tid:
#                     out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
#                     print(f"    LOSER: TEST_ID {tid} set to 0")
#
#     print(f"\n[RULE TWO] FLAGS UPDATED = {flags_updated}")
#     return out_df, ref_flag, flags_updated
#
# def rule_three(out_df, block_stats_dict, ref_flag):
#     print("\n[RULE THREE] Resolving multi-ref blocks with single test_id")
#
#     flags_updated = 0
#
#     for block, info in block_stats_dict.items():
#         block_size = info["block_size_ref_ids"]
#         count_test = info["count_test_ids"]
#
#         # Rule Three condition
#         if block_size > 1 and count_test == 1:
#             tid = info["test_ids"][0]
#             pred = float(info["pred_length"][tid])
#             ref_ids = info["ref_ids"]
#             actual_list = info["actual_length"][tid]   # list of candidate lengths
#
#             # Compute errors per ref
#             errors = {}
#             for rid, actual in zip(ref_ids, actual_list):
#                 errors[rid] = abs(pred - float(actual))
#
#             # Sort by best error
#             sorted_refs = sorted(errors, key=errors.get)
#
#             print(f"\nRULE3 BLOCK {block} TEST_ID={tid}")
#             print(f"    PRED={pred}")
#             print(f"    ERRORS={errors}")
#
#             assigned = False
#             for rid in sorted_refs:
#                 if ref_flag[rid] == 0:
#                     # Assign first available best ref
#                     out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = rid
#                     ref_flag[rid] = 1
#                     flags_updated += 1
#                     assigned = True
#                     print(f"    ASSIGNED REF {rid} to TEST_ID {tid}")
#                     break
#                 else:
#                     print(f"    REF {rid} already used, trying next...")
#
#             if not assigned:
#                 out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
#                 print(f"    NO AVAILABLE REF for TEST_ID {tid}, set to 0")
#
#     print(f"\n[RULE THREE] FLAGS UPDATED = {flags_updated}")
#     return out_df, ref_flag, flags_updated


# ================= RULE FUNCTIONS ================= #

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


def rule_two_single_block(out_df, info, ref_flag):
    flags_updated = 0
    ref_id = info["ref_ids"][0]

    if ref_flag[ref_id] == 1:
        print(f"RULE2 SKIP: REF {ref_id} already used")
        return out_df, ref_flag, 0

    errors = {}
    for tid in info["test_ids"]:
        pred = float(info["pred_length"][tid])
        actual = float(info["actual_length"][tid])
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


def rule_three_single_block(out_df, info, ref_flag):
    flags_updated = 0

    tid = info["test_ids"][0]
    pred = float(info["pred_length"][tid])
    ref_ids = info["ref_ids"]
    actuals = info["actual_length"][tid]

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


# DUMMY RULE FOUR
def rule_four_single_block(out_df, info, ref_flag):
    print("\n[RULE FOUR] Distance-aware multi-test multi-ref resolution")

    flags_updated = 0
    test_ids = info["test_ids"]
    ref_ids = info["ref_ids"]

    # Get distances
    distances = {}
    for tid in test_ids:
        distances[tid] = out_df.loc[out_df["id"] == tid, "distance_mm"].values[0]

    unique_distances = set(distances.values())

    # ================= CASE 1: ALL TEST IDS SAME DISTANCE ================= #
    if len(unique_distances) == 1:
        print(f"RULE4 CASE1: SAME DISTANCE {unique_distances.pop()} for TEST_IDS {test_ids}")

        # Compute min len_error per test_id
        best_test_error = {}

        for tid in test_ids:
            pred = float(info["pred_length"][tid])
            actuals = info["actual_length"][tid]

            min_err = min(abs(pred - float(a)) for a in actuals)
            best_test_error[tid] = min_err

        # Winner test_id
        winner_tid = min(best_test_error, key=best_test_error.get)
        print(f"RULE4 CASE1 WINNER TEST_ID={winner_tid} ERRORS={best_test_error}")

        # Pick best ref for winner
        pred = float(info["pred_length"][winner_tid])
        actuals = info["actual_length"][winner_tid]
        ref_errors = {rid: abs(pred - float(act)) for rid, act in zip(ref_ids, actuals)}
        sorted_refs = sorted(ref_errors, key=ref_errors.get)

        assigned = False
        for rid in sorted_refs:
            if ref_flag[rid] == 0:
                out_df.loc[out_df["id"] == winner_tid, "filtered_ids_new"] = rid
                ref_flag[rid] = 1
                flags_updated += 1
                assigned = True
                print(f"RULE4 CASE1 ASSIGNED REF {rid} TO TEST_ID {winner_tid}")
                break

        # Losers
        for tid in test_ids:
            if tid != winner_tid:
                out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
                print(f"RULE4 CASE1 LOSER TEST_ID {tid} -> 0")

        return out_df, ref_flag, flags_updated

    # ================= CASE 2: DIFFERENT DISTANCES ================= #
    print(f"RULE4 CASE2: DIFFERENT DISTANCES {distances}")

    # Sort test_ids by distance
    sorted_test_ids = sorted(test_ids, key=lambda t: distances[t])

    for tid in sorted_test_ids:
        pred = float(info["pred_length"][tid])
        actuals = info["actual_length"][tid]

        # Compute ref errors
        ref_errors = {rid: abs(pred - float(act)) for rid, act in zip(ref_ids, actuals)}
        sorted_refs = sorted(ref_errors, key=ref_errors.get)

        assigned = False
        for rid in sorted_refs:
            if ref_flag[rid] == 0:
                out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = rid
                ref_flag[rid] = 1
                flags_updated += 1
                assigned = True
                print(f"RULE4 CASE2 ASSIGNED TEST_ID={tid} -> REF={rid}")
                break

        if not assigned:
            out_df.loc[out_df["id"] == tid, "filtered_ids_new"] = 0
            print(f"RULE4 CASE2 NO REF AVAILABLE FOR TEST_ID {tid}")

    return out_df, ref_flag, flags_updated



# ================= RULE DISPATCHER ================= #

def rule_dispatcher(out_df, block_stats_dict, ref_flag):
    print("\n===== RULE DISPATCHER START =====")

    total_flags = 0

    for block, info in block_stats_dict.items():
        block_size = info["block_size_ref_ids"]
        count_test = info["count_test_ids"]

        print(f"\n--- BLOCK value = {block} | block_size_references = {block_size} | count_test_ids = {count_test} ---")

        # RULE ONE
        if block_size == 1 and count_test == 1:
            out_df, ref_flag, c = rule_one_single_block(out_df, info, ref_flag)

        # RULE TWO
        elif block_size == 1 and count_test > 1:
            out_df, ref_flag, c = rule_two_single_block(out_df, info, ref_flag)

        # RULE THREE
        elif block_size > 1 and count_test == 1:
            out_df, ref_flag, c = rule_three_single_block(out_df, info, ref_flag)

        # RULE FOUR (DUMMY)
        elif block_size > 1 and count_test > 1:
            out_df, ref_flag, c = rule_four_single_block(out_df, info, ref_flag)

        else:
            print("NO RULE MATCHED (should not happen)")
            c = 0

        total_flags += c

    print(f"\n===== TOTAL FLAGS UPDATED = {total_flags} =====")
    return out_df, ref_flag


# ================= MAIN MAKER FUNCTION ================= #

def filtered_ids_new_maker(out_df, block_stats_dict, max_ref_id):

    # Initialize ref flags
    ref_flag = {i: 0 for i in range(1, max_ref_id + 1)}
    print(f"\nREF FLAGS INITIALIZED (1..{max_ref_id})")

    # Initialize column
    out_df["filtered_ids_new"] = 0

    # Run dispatcher
    out_df, ref_flag = rule_dispatcher(out_df, block_stats_dict, ref_flag)

    # Final debug
    used_refs = [r for r, v in ref_flag.items() if v == 1]
    print("\nUSED REF IDS:", used_refs)
    print("TOTAL USED REF COUNT:", len(used_refs))

    return out_df, ref_flag
