import math
import numpy as np
from typing import List, Optional
from scipy.stats import truncnorm

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory, get_feature_weights
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from usePydl.predictors.helpers.tree import Tree
from src.data.data_obj.sampels import Samples
from src.samplers.single_gaussian import SingleGaussian1DSampler
import usePydl.predictors.helpers.interval as INTERVAL
import src.usePydl.predictors.helpers.groups as groups




class RandomForestPredictor:
    def __init__(self, sample_obj: Samples, n_trees=0,subsample_ratio: float = 0.7):
        self.trees: List[Tree] = []
        self.sample_obj = sample_obj
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.n_chunks = sample_obj.loader.n_chunks
        self.n_features = len(self.chunk_info.processed_feat_min)
        self.features_each_tree = self.n_features
        self.trees_each_chunk = 1
        self.subsample_ratio = subsample_ratio

        n_features_each_tree = int(math.ceil(np.sqrt(self.n_features)))
        if n_features_each_tree > self.n_features:
            self.features_each_tree = int(self.n_features)
            self.trees_each_chunk = 1
        else:
            self.features_each_tree = n_features_each_tree
            self.trees_each_chunk = max(4, int(np.ceil((self.n_features / self.features_each_tree) * 5)))
        self.n_trees = self.n_chunks * self.trees_each_chunk
        print(f"Init Random Forest: {self.n_trees} trees, each {self.features_each_tree} features")
        self.build_ensemble()


    def build_ensemble(self):
        for chunk_id in range(self.n_chunks):
            self.sample_obj.load_chunk(chunk_id)
            chunk_samples = self.sample_obj.samples
            n_total = chunk_samples.shape[0]
            n_sub = int(np.ceil(n_total * self.subsample_ratio))
            sub_indices = np.random.choice(n_total, size=n_sub, replace=False)
            sub_samples = chunk_samples[sub_indices]
            feature_groups = groups.create_total_feat_balanced_groups(n_groups=self.trees_each_chunk,
                                                                       features_each_group=self.features_each_tree,
                                                                       total_features=self.n_features)

            for i, group in enumerate(feature_groups):
                print(f"Tree num: {i}, (chunk:{chunk_id}, feat:{group}) TRAINING")
                feat_hist = FeatureHistory(sub_samples)
                weights = get_feature_weights(mode='random', focus_on=group)
                feat_hist.creat_splits(weights_each_feature=weights)
                __predictor = build_tree_iteratively(feat_hist, weights)
                complete_tree = feat_hist.get_complete_tree_memory_efficient()
                self.trees.append(complete_tree)
                print(f"Tree(chunk:{chunk_id}, feat:{group}) Trained")
"""
    def gen_new_data_guided(self, n = 100, reference_samples:np.ndarray = None, conf = 0.8):

        if reference_samples is None:
            self.sample_obj.load_chunk(0)
            reference_samples = self.sample_obj.samples

        n_ref = reference_samples.shape[0]
        indices_ref = np.random.default_rng().choice(n_ref, size=n, replace=True)
        gen_samples = np.zeros((n, self.n_features))
        for gen_id,ref_id in enumerate(indices_ref):
            intervals = []
            #loop over all trees
            for t in self.trees:
                p = t.let_sample_treverse_tree(reference_samples[ref_id])
                splits = p['splits']
                directions = p['directions']
                for j, d in enumerate(directions):
                    split_obj = splits[j]
                    if d == 'L':
                        intervals.append(split_obj.left_interval())
                    else: intervals.append(split_obj.right_interval())
            #sample values from all intervals
            for feat in range(reference_samples.shape[1]):
                combined_inter = [(True,self.chunk_info.processed_feat_min[feat], self.chunk_info.processed_feat_max[feat],True)]
                for inter in intervals:
                    inter = inter.get(feat, None)
                    if inter is not None:
                        combined_inter = INTERVAL.intersect_union_interval_lists(combined_inter, inter)

                for inter in combined_inter:
                    if INTERVAL.contains_value(inter, reference_samples[ref_id,feat]): gen_samples[gen_id,feat] = SingleGaussian1DSampler.sample_from_interval(interval=inter)[0]
        return gen_samples"""

def score_path(path, intervals_each_feature, path_error, rel_prob):

    def count_splits(path_dict) -> int:
        if not path_dict: return 0
        total = 0
        for value in path_dict.vals():
            try:
                total += len(value[0])
            except Exception:
                pass
        return int(total)

    n_splits = float(len(path['directions']))
    total_len = 0.0
    for feat_id in range(n_features):
        total_len += intervals_each_feature[feat_id].get_total_lenght()

    score = (n_splits + 1.0) * float(rel_prob)
    score /= ((1.0 + float(path_error)) * (1.0 + total_len))
    return float(score)


