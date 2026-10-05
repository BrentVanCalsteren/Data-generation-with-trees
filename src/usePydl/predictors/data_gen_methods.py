import numpy as np
from typing import List, Optional, Union, Dict

import CONFIG

import src.usePydl.predictors.helpers.interval as INTERVAL
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.uniform import UniformSampler
from usePydl.predictors.helpers.tree import Tree


#gen method 1
def gen_data_from_single_tree(complete_tree : Tree, n: int = 100, conf: float = 0.8) -> np.ndarray:
    intervals_each_path = complete_tree.create_intervals_each_paths()
    leafs = complete_tree.get_leafs()
    n_feats = len(CONFIG.GLOBAL_CHUNK_INFO.feature_importance)

    probs_each_path = np.array([l['value']['rel_prob'] for l in leafs])
    prob_sum = np.sum(probs_each_path)
    if prob_sum > 0:
        probs_each_path /= prob_sum
    else:
        probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)

    all_new_samples = np.array([])
    indx_list_n = np.random.choice(len(intervals_each_path), n, p=probs_each_path)
    indx, counts = np.unique(indx_list_n, return_counts=True)
    sample_count = 0

    for idx, count in zip(indx, counts):
        gen_feat_matrix = np.zeros((n_feats, count))
        intervals_each_feature = intervals_each_path[idx]
        sample_count += count
        global_min = CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min
        global_max = CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max
        for feat in range(n_feats):
            intervals = intervals_each_feature.get(feat, [(True, global_min[feat], global_max[feat], True)])
            if len(intervals) == 0:
                intervals = [(True, global_min[feat], global_max[feat], True)]
            gen_feat_matrix[feat] = SingleGaussian1DSampler.sample_from_interval_list_single_feature(intervals, count)

        if all_new_samples.size > 0: all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
        else: all_new_samples = gen_feat_matrix.T

    print(f"Generated {sample_count} samples.")
    if all_new_samples.size == 0: return all_new_samples
    return all_new_samples




#gen method 2
def gen_new_data_guided(predictor, n=100, reference_samples: np.ndarray = None, conf=0.8):
    if reference_samples is None:
        predictor.sample_obj.load_chunk(0)
        reference_samples = predictor.sample_obj.samples

    n_ref = reference_samples.shape[0]
    indices_ref = np.random.default_rng().choice(n_ref, size=n, replace=True)
    gen_samples = np.zeros((n, len(predictor.chunk_info.processed_feat_min)))
    for gen_id, ref_id in enumerate(indices_ref):
        intervals = []
        # loop over all trees
        for t in predictor.trees:
            intervals += t.get_intervals_based_on_sample(reference_samples[ref_id])
        # sample values from all intervals
        for feat in range(reference_samples.shape[1]):
            combined_inter = [(True, predictor.chunk_info.processed_feat_min[feat], predictor.chunk_info.processed_feat_max[feat], True)]
            for inter in intervals:
                if feat in inter:
                    combined_inter = INTERVAL.intersect_union_interval_lists(combined_inter, inter[feat])

            for inter in combined_inter:
                if INTERVAL.contains_value(inter, reference_samples[ref_id, feat]):
                    gen_samples[gen_id, feat] = SingleGaussian1DSampler.sample_from_interval_list_single_feature(interval_list=[inter])[0]
    return gen_samples


def map_p_trees_to_feat_interval_list(predictor):
    n_feats = CONFIG.GLOBAL_CHUNK_INFO.n_features
    global_min = np.array(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min, dtype=np.float64)
    global_max = np.array(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max, dtype=np.float64)

    tree_leaf_dic = {}

    from usePydl.predictors.ensemble_predictors import EnsemblePredictor
    if isinstance(predictor, EnsemblePredictor):
        trees = predictor.trees
    else: trees = [predictor.tree]

    #for each tree extract all leaf_ids
    for t_idx, tree in enumerate(trees):
        path_intervals = tree.create_intervals_each_paths()
        leafs = tree.get_leafs()
        leaf_dic = {}
        for leaf in leafs:
            l_id = leaf['value']['leaf_id']
            weight = float(leaf['value']['rel_prob'])
            path_dict = path_intervals.get(l_id)

            #pat unsplit feat with global_min or global_max
            leaf_feats = []
            for f_idx in range(n_feats):
                iv_list = path_dict.get(f_idx)
                if iv_list: leaf_feats.append(iv_list)
                else: leaf_feats.append([(True, global_min[f_idx], global_max[f_idx], True)])
            leaf_dic[l_id] = leaf_feats
        tree_leaf_dic[t_idx] = leaf_dic

    #convert inner dic to 2d list
    for tree_id ,leaf_dic in tree_leaf_dic.items():
        compact_list = []
        leaf_ids = list(leaf_dic.keys())
        feat_list = list(leaf_dic.values())
        for i, intervals_of_feat in enumerate(feat_list):
            leaf_id = leaf_ids[i]
            for feat_id ,intervals in enumerate(intervals_of_feat):
                for intval in intervals:
                    compact_list.append([feat_id,leaf_id,intval])
        tree_leaf_dic[tree_id] = np.array(compact_list, dtype=object)
    return tree_leaf_dic


def generate_intersect_value_for_feat_from_trees_dic(trees_dic,feat_id,n_samples,prev_generated_features=None):

    rng = np.random.default_rng()

    # 1. Pre-process trees: group intervals by path/leaf ID for fast lookup
    # Each entry: dict of {path_id: {feat: interval}}
    tree_paths = []
    for tree_intervals in trees_dic.values():
        paths = {}
        # Assumes structure: [feat_id, path_id, interval]
        # If your array only has [feat_id, interval], path_id is the row index
        has_path_col = tree_intervals.shape[1] >= 3
        for row in tree_intervals:
            f = row[0]
            p_id = row[1] if has_path_col else id(row)
            interval = row[2] if has_path_col else row[1]

            if p_id not in paths:
                paths[p_id] = {}
            paths[p_id][f] = interval
        tree_paths.append(paths)

    n_trees = len(tree_paths)
    samples = np.full(n_samples, np.nan)

    # Helper: Check if a scalar value falls within an interval
    def value_in_interval(val, interval):
        # Supports [low, high] bounds, tuple, or an INTERVAL object with a .contains() method
        if hasattr(interval, "contains"):
            return interval.contains(val)
        if isinstance(interval, (list, tuple, np.ndarray)) and len(interval) == 2:
            return interval[0] <= val <= interval[1]
        # Fallback to intersect check if interval is an object list
        return len(INTERVAL.intersect_union_interval_lists([interval], [(True,val, val,True)])) > 0

    for i in range(n_samples):
        # 2. Identify active previous features for sample i (rows that are not NaN)
        if prev_generated_features is not None:
            col_vals = prev_generated_features[:, i]
            active_feats = [
                (f_idx, col_vals[f_idx])
                for f_idx in range(len(col_vals))
                if not np.isnan(col_vals[f_idx])
            ]
        else:
            active_feats = []

        # 3. Filter paths in each tree that:
        #    a) Contain a constraint for feat_id
        #    b) Accept all previously generated feature values for sample i
        valid_tree_target_intervals = []
        for paths in tree_paths:
            candidate_target_intervals = []
            for p_id, feat_map in paths.items():
                if feat_id not in feat_map:
                    continue

                # Check consistency against all previously generated features
                path_compatible = True
                for prev_f, prev_val in active_feats:
                    if prev_f in feat_map:
                        if not value_in_interval(prev_val, feat_map[prev_f]):
                            path_compatible = False
                            break

                if path_compatible:
                    candidate_target_intervals.append(feat_map[feat_id])

            if candidate_target_intervals:
                valid_tree_target_intervals.append(candidate_target_intervals)

        if not valid_tree_target_intervals:
            continue

        # 4. Shuffle tree order and perform sequential intersection
        n_valid = len(valid_tree_target_intervals)
        tree_order = rng.permutation(n_valid)

        first_tree = valid_tree_target_intervals[tree_order[0]]
        picked_idx = rng.integers(len(first_tree))
        hoofd_intervals = [first_tree[picked_idx]]

        for t_idx in tree_order[1:]:
            cand_tree = valid_tree_target_intervals[t_idx]
            cand_idx = rng.integers(len(cand_tree))
            interval2 = [cand_tree[cand_idx]]

            combo = INTERVAL.intersect_union_interval_lists(hoofd_intervals, interval2)
            if len(combo) == 0:
                break
            hoofd_intervals = combo

        if len(hoofd_intervals) > 0:
            samples[i] = UniformSampler.sample_from_interval_list_single_feature(hoofd_intervals)[0]

    return samples


#gen method 3
def gen_new_data_by_chain_feat_sampling(predictor,n = 100,n_refine_sweeps= 0,sample_mode = 'uniform'):
    n_feat = CONFIG.GLOBAL_CHUNK_INFO.n_features
    if not predictor.trees:
        return np.zeros((n, n_feat), dtype=np.float64)

    tree_interval_dic = map_p_trees_to_feat_interval_list(predictor)

    new_samples = np.full((n_feat, n), np.nan)
    for j in range(n_feat):
        new_sample_array = generate_intersect_value_for_feat_from_trees_dic(tree_interval_dic, feat_id=j,n_samples=n,prev_generated_features=new_samples)
        new_samples[j,:] = new_sample_array
        print(f'GENERATED NEW SAMPLE ARRAY: index{j}')
    return new_samples.T




def gen_new_data_gibbs(predictor,n: int = 100,n_sweeps: int = 4,strategy: str = 'consensus',
    sample_mode: str = 'uniform',initial_samples: Optional[np.ndarray] = None):
    n_feats = CONFIG.GLOBAL_CHUNK_INFO.n_features
    global_min = np.array(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min, dtype=np.float64)
    global_max = np.array(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max, dtype=np.float64)

    if not predictor.trees:
        return np.zeros((n, n_feats), dtype=np.float64)

    # 1. Feature Sequencing Order: Feature Importance Order
    feat_imp = getattr(CONFIG.GLOBAL_CHUNK_INFO, 'feature_importance', None)
    if feat_imp is not None and len(feat_imp) == n_feats:
        feat_order = np.argsort(-np.asarray(feat_imp, dtype=float)).tolist()
    else:
        feat_order = list(range(n_feats))

    # 2. Precompute path intervals with Global Min/Max Padding for unsplit features
    leaf_tree_ids = []
    leaf_weights = []
    leaf_intervals = []

    has_tree_w = hasattr(predictor, 'tree_weights') and len(predictor.tree_weights) == len(predictor.trees)

    for t_idx, tree in enumerate(predictor.trees):
        path_intervals = tree.create_intervals_each_paths()
        tw = predictor.tree_weights[t_idx] if has_tree_w else 1.0
        for leaf in tree.get_leafs():
            l_id = leaf['value']['leaf_id']
            weight = float(leaf['value']['rel_prob']) * tw
            path_dict = path_intervals.get(l_id, {})

            leaf_tree_ids.append(t_idx)
            leaf_weights.append(weight)

            # Pad unsplit features with [global_min, global_max]
            leaf_dict = {}
            for f_idx in range(n_feats):
                iv_list = path_dict.get(f_idx, [])
                if iv_list:
                    leaf_dict[f_idx] = [(iv[1], iv[2]) for iv in iv_list]
                else:
                    leaf_dict[f_idx] = [(global_min[f_idx], global_max[f_idx])]
            leaf_intervals.append(leaf_dict)

    n_leaves = len(leaf_tree_ids)
    if n_leaves == 0:
        return np.random.uniform(global_min, global_max, size=(n, n_feats))

    leaf_tree_ids = np.array(leaf_tree_ids, dtype=np.int32)
    leaf_weights = np.array(leaf_weights, dtype=np.float64)
    n_trees = len(predictor.trees)

    max_iv = 1
    for ld in leaf_intervals:
        for ivs in ld.values():
            if len(ivs) > max_iv:
                max_iv = len(ivs)

    np_low = np.full((n_leaves, n_feats, max_iv), np.nan, dtype=np.float64)
    np_high = np.full((n_leaves, n_feats, max_iv), np.nan, dtype=np.float64)
    np_valid = np.zeros((n_leaves, n_feats, max_iv), dtype=bool)
    np_is_split = np.zeros((n_leaves, n_feats), dtype=bool)

    for l_idx, ld in enumerate(leaf_intervals):
        for f_idx in range(n_feats):
            ivs = ld[f_idx]
            is_actual_split = not (len(ivs) == 1 and np.isclose(ivs[0][0], global_min[f_idx]) and np.isclose(ivs[0][1], global_max[f_idx]))
            np_is_split[l_idx, f_idx] = is_actual_split
            for k, (low, high) in enumerate(ivs):
                np_low[l_idx, f_idx, k] = low
                np_high[l_idx, f_idx, k] = high
                np_valid[l_idx, f_idx, k] = True

    # Helper: Density-weighted sampling within a consensus union
    def sample_from_consensus(consensus_ivs, f_idx, active_mask, mode='uniform'):
        if not consensus_ivs:
            return np.random.uniform(global_min[f_idx], global_max[f_idx])

        # Density weighting: score each interval piece by active leaf support * width
        if len(consensus_ivs) > 1:
            scores = []
            for iv in consensus_ivs:
                mid_pt = (iv[1] + iv[2]) / 2.0
                in_mid = (mid_pt >= (np_low[:, f_idx, :] - 1e-9)) & (mid_pt <= (np_high[:, f_idx, :] + 1e-9)) & np_valid[:, f_idx, :]
                supp = leaf_weights[active_mask & in_mid.any(axis=-1)].sum() if active_mask.any() else 1.0
                width = max(1e-9, iv[2] - iv[1])
                scores.append(max(1e-9, supp) * width)
            scores = np.array(scores, dtype=np.float64)
            s_sum = scores.sum()
            p_scores = scores / s_sum if s_sum > 0 else np.ones(len(scores)) / len(scores)
            chosen_idx = np.random.choice(len(consensus_ivs), p=p_scores)
            chosen_iv = consensus_ivs[chosen_idx]
        else:
            chosen_iv = consensus_ivs[0]

        low_b, high_b = chosen_iv[1], chosen_iv[2]
        if high_b <= low_b:
            return low_b

        if mode in ('gaussian', 'sharp'):
            mid = (low_b + high_b) / 2.0
            span = (high_b - low_b) / 2.0
            z = np.clip(np.random.normal(0, 0.40), -1.0, 1.0)
            return mid + z * span
        else:
            return np.random.uniform(low_b, high_b)

    # Helper: Multi-tree consensus intersection on feature f
    def get_tree_consensus(f_idx, active_mask, strat='consensus'):
        has_split = active_mask & np_is_split[:, f_idx]
        split_trees = np.unique(leaf_tree_ids[has_split]) if has_split.any() else np.array([], dtype=int)
        target_trees = split_trees if len(split_trees) > 0 else np.unique(leaf_tree_ids[active_mask])

        tree_proposals = []
        for t in target_trees:
            t_cand = np.where(active_mask & (leaf_tree_ids == t))[0]
            if len(t_cand) == 0:
                continue
            t_weight = leaf_weights[t_cand].sum()

            t_iv_list = []
            for l_idx in t_cand:
                for k in range(max_iv):
                    if np_valid[l_idx, f_idx, k]:
                        t_iv_list.append((True, np_low[l_idx, f_idx, k], np_high[l_idx, f_idx, k], True))
            t_union = INTERVAL.merge_intervals(t_iv_list)
            if t_union:
                tree_proposals.append((t_weight, t_union))

        if not tree_proposals:
            return [(True, global_min[f_idx], global_max[f_idx], True)]

        # Order trees by active support descending
        tree_proposals.sort(key=lambda x: x[0], reverse=True)

        if strat == 'max_weight':
            consensus_ivs = tree_proposals[0][1]
            for _, t_union in tree_proposals[1:]:
                res = INTERVAL.intersect_union_interval_lists(consensus_ivs, t_union)
                if len(res) > 0:
                    consensus_ivs = res
            return consensus_ivs
        else:
            consensus_ivs = [(True, global_min[f_idx], global_max[f_idx], True)]
            for _, t_union in tree_proposals:
                res = INTERVAL.intersect_union_interval_lists(consensus_ivs, t_union)
                if len(res) > 0:
                    consensus_ivs = res
            return consensus_ivs

    # 3. Warm Start: Initialize samples in valid clusters
    if initial_samples is not None:
        current_samples = np.array(initial_samples, dtype=np.float64, copy=True)
    else:
        # Use chain feature sampling for a robust, multi-tree cluster warm-start
        current_samples = gen_new_data_by_chain_feat_sampling(
            predictor=predictor,
            n=n,
            n_refine_sweeps=0,
            sample_mode=sample_mode
        )

    # 4. Gibbs Sweeps
    for sweep in range(n_sweeps):
        for f in feat_order:
            for i in range(n):
                cur_x = current_samples[i]

                # Identify leaves compatible with cur_x on all features EXCEPT f
                in_bounds = ((cur_x[None, :, None] >= (np_low - 1e-7)) &
                             (cur_x[None, :, None] <= (np_high + 1e-7)) &
                             np_valid)
                match_feat = (~np_is_split) | in_bounds.any(axis=-1)

                active_for_f = np.delete(match_feat, f, axis=1).all(axis=1)

                # If there are active leaves constraining f, update coordinate
                if (active_for_f & np_is_split[:, f]).any():
                    refine_consensus = get_tree_consensus(f, active_for_f, strat=strategy)
                    if refine_consensus:
                        new_val = sample_from_consensus(refine_consensus, f, active_for_f, mode=sample_mode)
                        current_samples[i, f] = new_val

    return current_samples


def find_all_intervals_that_accept_partial_sample(partial_sample, trees):
    all_intervals = []

    for tree in trees:
        intervals_dic = tree.create_intervals_each_paths()
        for interval_dic in list(intervals_dic.values()):
            valid_interval = True
            for i in range(len(partial_sample)):
                if not INTERVAL.contains_in_dict(interval_dic, partial_sample[i], i): valid_interval = False
            if valid_interval: all_intervals.append(interval_dic)
    return all_intervals

def remove_invalid_intervals(partial_sample, intervals: List):
    all_intervals = []
    for interval_dic in intervals:
        valid_interval = True
        for i in range(len(partial_sample)):
            if not INTERVAL.contains_in_dict(interval_dic, partial_sample[i], i): valid_interval = False
        if valid_interval: all_intervals.append(interval_dic)
    return all_intervals

def keep_intervals_with_smallest_domain(il_1, il_2):
    dist_1 = 0
    for interval in il_1: dist_1 += interval[2] - interval[1]
    dist_2 = 0
    for interval in il_2: dist_2 += interval[2] - interval[1]
    if dist_1 < dist_2: return il_1
    else: return il_2