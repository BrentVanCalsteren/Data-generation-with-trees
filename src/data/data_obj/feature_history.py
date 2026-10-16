import copy

import math
from typing import List

import numpy as np

import CONFIG
from splits.splits import Splits
from trees.predictors.helpers.tree import Tree, remap_tree, extend_tree


def create_new_history(samples):
    feat_history = FeatureHistory(samples=samples)
    return feat_history

def extend_history(new_samples, old_history: FeatureHistory, l_id):
    new_history = FeatureHistory(samples=new_samples)
    old_history.add_future(new_history)
    new_history.set_past(old_history)
    new_history.leaf_id = l_id
    new_history.depth = old_history.depth + 1
    return new_history



class FeatureHistory:
    def __init__(self, samples):
        self.samples = samples
        self.past = None
        self.future = []
        self.splits_obj: Splits = None
        self.depth = 0
        self.tree:Tree = None
        self.complete_tree:Tree = None
        self.leaf_id = None
        self.pred_error = np.inf

    def get_complete_tree_memory_efficient(self):
        self.reduce_memory()
        if self.complete_tree is not None: complete_tree = self.complete_tree
        else: complete_tree = self.get_complete_tree()
        return complete_tree

    def reduce_memory(self):
        self.complete_tree = self.get_complete_tree()
        self.tree = None
        self.samples = None
        self.splits_obj = None
        self.past = None
        self.future = []

    def set_past(self,history: FeatureHistory):
        self.past = history

    def add_future(self,history: FeatureHistory):
        self.future.append(history)

    def get_first_hist_depth_0(self):
        hist = self
        depth = hist.depth
        while depth > 0:
            hist = hist.past
            depth = hist.depth
        return hist

    def get_sample_array_from_history(self):
        return self.samples


    def creat_splits(self, weights_each_feature=None):
        n_feats = self.samples.shape[1]

        num_splits_featurs = []
        MAX_SPLITS = n_feats * CONFIG.AVG_BOOL_SPLITS_EACH_FEATURE
        if weights_each_feature is not None:
            #generate splits bassed on weight
            for i in range(n_feats):
                w = weights_each_feature[i]
                num_splits_of_feat = math.ceil(MAX_SPLITS * w)
                num_splits_featurs.append(num_splits_of_feat)

        features = self.samples.T
        self.splits_obj = Splits(features=features, max_splits_each_feature=num_splits_featurs, weights_each_feature=weights_each_feature, max_splits=MAX_SPLITS)


    def get_splits_array(self):
        return self.splits_obj.get_splits_array()

    def store_split_data(self, idx):
        return self.splits_obj.all_data_on_id(idx)

    def remove_unused_split_data(self, used_ids):
        self.splits_obj.remove_unused_split_data(used_ids)

    def get_local_tree(self):
        return self.tree

    def get_complete_tree(self):
        root = self
        futures = root.future

        if self.tree is None:
            if self.complete_tree is not None: return copy.deepcopy(self.complete_tree)
            else:
                print('no tree obj, prob reduced mem without saving complete tree')
                return None

        tree = copy.deepcopy(self.tree.tree)
        if tree is None: return None
        tree = remap_tree(tree, feature_history=self)
        while len(futures) > 0:
            new_futures = []
            for hist in futures:
                if hist.tree is not None:
                    if hist.tree.tree is not None:
                        hist_tree = copy.deepcopy(hist.tree.tree)
                        hist_tree = remap_tree(hist_tree, feature_history=hist)
                        tree = extend_tree(tree, hist_tree, hist.leaf_id)
                        new_futures += hist.future
                    else: print('no tree dict')
                else: print('no tree obj')
            futures = new_futures
        return Tree(tree=tree)

def get_feature_weights( mode: str = 'uniform', focus_on: List[int]=None):
    n_feats = len(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min)
    weights = np.ones(n_feats, dtype=float) / float(n_feats) #uniform

    if n_feats == 0: return np.array([])
    if mode == 'random': weights = np.random.uniform(0.1, 1.0, size=n_feats)
    elif mode == 'uniform': pass
    elif mode == 'feat_imp' and CONFIG.GLOBAL_CHUNK_INFO.feature_importance is not None: weights = np.asarray(CONFIG.GLOBAL_CHUNK_INFO.feature_importance).flatten()
    else: return None

    if focus_on is not None and len(focus_on) > 0:
        mask = np.zeros(n_feats, dtype=bool)
        mask[focus_on] = True
        weights[~mask] = 0.0

    weights = weights / np.sum(weights) #normilze
    return weights

