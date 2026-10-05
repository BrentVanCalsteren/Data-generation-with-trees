import numpy as np
from pydl85 import DL85Predictor

import CONFIG
from src.usePydl.error_fun import IntervalSizesError, ClusterError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.single_gaussian import SingleGaussian1DSampler
from usePydl.predictors.helpers.interval import Interval, intersect_union_interval_lists

from usePydl.predictors.helpers.tree import Tree, remap_tree

class Predictor:
    def __init__(self, sub_samples,n_total_samples,splits, weights, passed_error_fun=None,use_c_error='cluster_error'):
        print('starting dl predictor...')

        if weights is None:
            weights = [1]*sub_samples.shape[1]

        c_error = None
        if passed_error_fun is not None:
            error_fun = passed_error_fun
        elif use_c_error is not None:
            error_fun = None
            c_error = use_c_error
        else: error_fun = ClusterError(samples=sub_samples)
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
            costum_error_type=c_error   #"interval_error", "cluster_error", "mse_error"
                                          )
        self.dl_predictor.fit(splits.T)
        self.error = self.dl_predictor.error_
        self.tree = Tree(tree=self.dl_predictor.tree_)
        self.dl_predictor = None


