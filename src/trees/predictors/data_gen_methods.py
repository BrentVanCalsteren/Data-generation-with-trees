import numpy as np
from typing import List, Optional, Union, Dict

import CONFIG

import src.trees.predictors.helpers.interval as INTERVAL
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.uniform import UniformSampler
from trees.predictors.helpers.tree import Tree


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

    from trees.predictors.ensemble_predictors import EnsemblePredictor
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
            #weight = float(leaf['value']['rel_prob'])
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
    #dict of {path_id: {feat: interval}}
    tree_paths = []
    for tree_intervals in trees_dic.values():
        paths = {}
        has_path_col = tree_intervals.shape[1] >= 3
        for row in tree_intervals:
            f = row[0]
            p_id = row[1] if has_path_col else id(row)
            interval = row[2] if has_path_col else row[1]

            if p_id not in paths: paths[p_id] = {}
            paths[p_id][f] = interval
        tree_paths.append(paths)

    n_trees = len(tree_paths)
    new_samples = np.full(n_samples, np.nan)

    # Helper: Check if a scalar value falls within an interval
    def value_in_interval(val, interval):
        if isinstance(interval, (list, tuple, np.ndarray)) and len(interval) == 2:
            return interval[1] <= val <= interval[2]
        # Fallback to intersect check if interval is an object list
        return len(INTERVAL.intersect_union_interval_lists([interval], [(True,val, val,True)])) > 0

    for i in range(n_samples):
        if prev_generated_features is not None:
            col_vals = prev_generated_features[:, i]
            active_feats = [(f_idx, col_vals[f_idx]) for f_idx in range(len(col_vals)) if not np.isnan(col_vals[f_idx])]
        else:
            active_feats = []

        valid_tree_target_intervals = []
        for paths in tree_paths:
            candidate_target_intervals = []
            for p_id, feat_map in paths.items():
                if feat_id not in feat_map: continue
                #conflict check
                conflict = False
                for prev_f, prev_val in active_feats:
                    if prev_f in feat_map:
                        if not INTERVAL.contains_in_dict(feat_map, prev_val, prev_f):
                            conflict = True
                            break

                if not conflict: candidate_target_intervals.append(feat_map[feat_id])

            if candidate_target_intervals: valid_tree_target_intervals.append(candidate_target_intervals)
        if not valid_tree_target_intervals: continue

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
            if len(combo) == 0: break
            hoofd_intervals = combo

        if len(hoofd_intervals) > 0: new_samples[i] = UniformSampler.sample_from_interval_list_single_feature(hoofd_intervals)[0]
    return new_samples


#gen method 3
def gen_new_data_by_chain_feat_sampling(predictor,n = 100):
    n_feat = CONFIG.GLOBAL_CHUNK_INFO.n_features
    tree_interval_dic = map_p_trees_to_feat_interval_list(predictor)

    new_samples = np.full((n_feat, n), np.nan)
    for j in range(n_feat):
        new_sample_array = generate_intersect_value_for_feat_from_trees_dic(tree_interval_dic, feat_id=j,n_samples=n,prev_generated_features=new_samples)
        new_samples[j,:] = new_sample_array
        print(f'GENERATED NEW SAMPLE ARRAY: index{j}')
    return new_samples.T


def gen_new_data_gibbs(predictor,n: int = 100,n_sweeps: int = 4,sample_mode: str = 'uniform') -> np.ndarray:
    n_feat = CONFIG.GLOBAL_CHUNK_INFO.n_features
    global_min = np.asarray(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min, dtype=np.float64)
    global_max = np.asarray(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max, dtype=np.float64)

    tree_interval_dic = map_p_trees_to_feat_interval_list(predictor)

    tree_leaves = {}
    all_leaves = []
    for tree_id, feat_leaf_intervals in tree_interval_dic.items():
        tree_leaves[tree_id] = {}
        for feat_id, leaf_id, intval in feat_leaf_intervals:
            if leaf_id not in tree_leaves[tree_id]: tree_leaves[tree_id][leaf_id] = {}
            tree_leaves[tree_id][leaf_id][feat_id] = (float(intval[1]), float(intval[2]), intval)
        all_leaves.extend(list(tree_leaves[tree_id].values()))

    new_samples = np.full((n_feat, n), np.nan, dtype=np.float64)
    for i in range(n):
        chosen_leaf = all_leaves[np.random.randint(len(all_leaves))] if all_leaves else {}
        for f in range(n_feat):
            if f in chosen_leaf: low, high = chosen_leaf[f][0], chosen_leaf[f][1]
            else: low, high = global_min[f], global_max[f]

            new_samples[f, i] = np.random.uniform(low, high)

    for sweep in range(n_sweeps):
        for f in range(n_feat):
            for i in range(n):
                cur_x = new_samples[:, i]
                consensus = [(True, global_min[f], global_max[f], True)]

                # Collect valid intervals on feature f across all trees
                for tree_id, leaves in tree_leaves.items():
                    tree_f_intervals = []

                    for leaf_id, feat_bounds in leaves.items():
                        # Leaf matches if cur_x falls within bounds for all coordinates j != f
                        compatible = True
                        for j, (l_bound, u_bound, _) in feat_bounds.items():
                            if j != f:
                                if not (l_bound - 1e-7 <= cur_x[j] <= u_bound + 1e-7):
                                    compatible = False
                                    break

                        if compatible:
                            # If leaf constrains feature f, use it; otherwise use global range
                            if f in feat_bounds:
                                tree_f_intervals.append(feat_bounds[f][2])
                            else:
                                tree_f_intervals.append((True, global_min[f], global_max[f], True))

                    if not tree_f_intervals:
                        continue

                    # Merge overlapping leaf intervals for this tree (if helper exists)
                    if hasattr(INTERVAL, 'merge_intervals') and len(tree_f_intervals) > 1:
                        tree_f_intervals = INTERVAL.merge_intervals(tree_f_intervals)

                    # Intersect this tree's proposals with running consensus
                    intersected = INTERVAL.intersect_union_interval_lists(consensus, tree_f_intervals)
                    if intersected:
                        consensus = intersected

                # 4. Sample coordinate f from the consensus interval
                if not consensus:
                    low_b, high_b = global_min[f], global_max[f]
                elif len(consensus) == 1:
                    low_b, high_b = consensus[0][1], consensus[0][2]
                else:
                    # Pick an interval piece weighted by length
                    widths = np.array([max(1e-9, iv[2] - iv[1]) for iv in consensus], dtype=np.float64)
                    probs = widths / widths.sum() if widths.sum() > 0 else None
                    chosen_iv = consensus[np.random.choice(len(consensus), p=probs)]
                    low_b, high_b = chosen_iv[1], chosen_iv[2]

                if high_b <= low_b:
                    new_samples[f, i] = low_b
                elif sample_mode == 'uniform':
                    new_samples[f, i] = np.random.uniform(low_b, high_b)
                else:
                    mid = (low_b + high_b) / 2.0
                    span = (high_b - low_b) / 2.0
                    z = np.clip(np.random.normal(0, 0.40), -1.0, 1.0)
                    new_samples[f, i] = np.clip(mid + z * span, low_b, high_b)
    return new_samples.T


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