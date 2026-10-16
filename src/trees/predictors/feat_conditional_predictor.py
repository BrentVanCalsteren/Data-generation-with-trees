from typing import List, Optional, Dict, Union
import numpy as np

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory, get_feature_weights
from src.trees.predictors.predictor import Predictor
from src.trees.predictors.greedy_deepening_predictor import build_tree_iteratively
from src.trees.error_fun import ClusterError, chainTreeError
from src.trees.predictors.helpers.tree import Tree
from src.samplers.single_gaussian import SingleGaussian1DSampler
import src.trees.predictors.helpers.interval as INTERVAL
from trees.predictors.helpers.tree import Tree


class FeatConditionalPredictor:
    #works entirly from the idea that P(x0,x1,..,xn) = P(x0) * P(x1|x0) * ... * P(xn|xn-1,...)
    def __init__(self,sample_obj=None, mode='random_forest'):
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.sample_obj = sample_obj
        self.n_feat = len(self.chunk_info.processed_feat_min)

        if self.chunk_info is None or self.chunk_info.feature_importance is None: self.feature_order = list(range(self.n_feat))
        else: self.feature_order = np.argsort(-self.chunk_info.feature_importance).tolist()
        self.trees = []
        self.build_tree_feat_chain()

    def build_tree_feat_chain(self):
        print(f"Building Feat cond Tree chain ({self.n_feat} features)...")

        for step, target_feat in enumerate(self.feature_order):
            cond_features = self.feature_order[:step]
            print(f"Training Tree for Feature {target_feat} (conditioned on {cond_features})")
            self.build_tree(target_feat, step)



    def build_tree(self, target_feat: int, step: int = 0):
        self.sample_obj.load_chunk(0)
        samples = self.sample_obj.samples
        error_fun = ClusterError(samples)
        error_fun = chainTreeError(error_fun,self.trees, step)
        feat_hist = FeatureHistory(samples=samples)
        weights = get_feature_weights(mode='uniform', focus_on=[target_feat])
        feat_hist.creat_splits(weights_each_feature=weights)
        splits_arr = feat_hist.get_splits_array()
        #p = Predictor(sub_samples=samples,n_total_samples=samples.shape[0],splits=splits_arr,weights=weights, passed_error_fun=error_fun)
        p = build_tree_iteratively(feature_history=feat_hist,weights=weights,passed_error_fun=error_fun)
        feat_hist.tree = p.tree
        complete_tree = feat_hist.get_complete_tree_memory_efficient()
        self.trees.append(complete_tree)

