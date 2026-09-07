import numpy as np
from pydl85 import DL85Predictor

import CONFIG
from src.usePydl.error_fun import IntervalSizesError, ClusterError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.single_gaussian import SingleGaussian1DSampler
from usePydl.predictors.helpers.interval import Interval, intersect_union_interval_lists

from usePydl.predictors.helpers.tree import Tree, remap_tree

class Predictor:
    def __init__(self, sub_samples,n_total_samples,splits, weights, use_c_error=True):
        print('starting dl predictor...')

        if weights is None:
            weights = [1]*sub_samples.shape[1]

        error_fun = None
        if not use_c_error:
            error_fun = ClusterError(samples=sub_samples,feature_weights=weights)
        leaf_val = ReturnIDSandPROB(n_total_samples)

        self.dl_predictor = DL85Predictor(
            error_function=error_fun,
            leaf_value_function=leaf_val,
            max_depth=CONFIG.MAX_DEPTH,
            min_sup=1,
            time_limit=CONFIG.MAX_TIME_PREDICTOR,
            max_error=np.inf,
            samples=sub_samples,
            feature_weights=weights,
            costum_error_type="cluster_error"   #"interval_error", "cluster_error", "mse_error"
                                          )
        self.dl_predictor.fit(splits.T)
        self.error = self.dl_predictor.error_
        self.tree = Tree(tree=self.dl_predictor.tree_)
        self.dl_predictor = None



def gen_data_from_single_tree(complete_tree : Tree, n: int = 100, conf: float = 0.8) -> np.ndarray:
    paths = complete_tree.get_all_paths()
    intervals_each_path = get_intervals_from_paths(paths)
    n_feats = len(CONFIG.GLOBAL_CHUNK_INFO.feature_importance)

    #path has 'directions', 'leaf_vals' and 'splits'
    if len(paths) == 0: raise ValueError("Tree has no paths")

    probs_each_path = np.array([path['leaf_vals']['rel_prob'] for path in paths])
    prob_sum = np.sum(probs_each_path)
    if prob_sum > 0:
        probs_each_path /= prob_sum
    else:
        probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)

    all_new_samples = np.array([])
    indx_list_n = np.random.choice(len(paths), n, p=probs_each_path)
    indx, counts = np.unique(indx_list_n, return_counts=True)
    sample_count = 0

    for idx, count in zip(indx, counts):
        gen_feat_matrix = np.zeros((n_feats, count))
        intervals_each_feature = intervals_each_path[idx]
        sample_count += count
        for feat in range(n_feats):
            intervals = intervals_each_feature[feat]
            if len(intervals) == 0: print('!THIS SHOULD NOT HAPPEN!')
            gen_feat_matrix[feat] = SingleGaussian1DSampler.sample_from_interval(intervals, count)

        if all_new_samples.size > 0: all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
        else: all_new_samples = gen_feat_matrix.T

    print(f"Generated {sample_count} samples.")
    if all_new_samples.size == 0: return all_new_samples
    return all_new_samples


def get_intervals_from_paths(paths):
    chunk_info = CONFIG.GLOBAL_CHUNK_INFO
    paths_intervals_feats = []
    for path in paths:

        splits = path['splits']
        directions = path['directions']
        intervals = []
        for j, d in enumerate(directions):
            split_obj = splits[j]
            if d == 'L': intervals.append(split_obj.left_interval())
            else: intervals.append(split_obj.right_interval())

        feat_inter = {}
        paths_intervals_feats.append(feat_inter)
        for feat in range(len(chunk_info.feature_importance)):
            combined_inter = [(True,chunk_info.processed_feat_min[feat], chunk_info.processed_feat_max[feat], True)]
            for inter in intervals:
                inter = inter.get(feat, None)
                if inter is not None: combined_inter = intersect_union_interval_lists(combined_inter, inter)
            feat_inter[feat] = combined_inter

    return paths_intervals_feats

