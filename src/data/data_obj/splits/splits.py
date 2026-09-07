import re
from typing import List

import CONFIG
from src.usePydl.error_fun import IntervalSizesError, MSEError, ClusterError
import numpy as np
from src.data.find_largest_gaps import DivisiveCluster
from scipy.stats import gaussian_kde
from src.data.data_obj.splits.split_type import SplitType
from src.data.data_obj.splits.bigger_split import BiggerEqSplit
from src.data.data_obj.splits.equal_split import EqualSplit
from src.data.data_obj.splits.interval_split import IntervalSplit

KEYS = ['types', 'values', 'feats', 'splits']

class Splits:

    def __init__(self, features,max_splits, max_splits_each_feature=None, weights_each_feature=None):
        samples = features.T
        self.max_splits_each_feature = max_splits_each_feature
        self.max_splits = max_splits
        self.weights_each_feature = weights_each_feature
        self.splits_dic = {}
        self.interval_error = IntervalSizesError(samples=samples,feature_weights=weights_each_feature)
        self.mse_error = MSEError(samples=samples,feature_weights=weights_each_feature)
        self.cluster_error = ClusterError(samples=samples,feature_weights=weights_each_feature)
        self.create_best_splits(features)


    def get_splits_array(self):
        return np.array(self.splits_dic['splits'])

    def all_data_on_id(self, idx):
        data = {}
        for key in ['feats']:
            data[key] = self.splits_dic[key][idx]
        data['left_interval'] = self.left_interval(idx)
        data['right_interval'] = self.right_interval(idx)
        return data

    def remove_unused_split_data(self, used_ids):
        clean_split_dic = SplitType.create_empty_split_directory()
        for idx in used_ids:
            for i in range(len(self.splits_dic['splits'])):
                if i == idx:
                    for key in KEYS:
                        clean_split_dic[key].append(self.splits_dic[key][i])
        self.splits_dic = clean_split_dic




    def create_best_splits(self, features):
        all_candidates = _generate_splits(self.max_splits_each_feature, features)
        all_splits = np.asarray(all_candidates['splits'])
        scores = self.score_splits(all_splits)
        best_indices = np.argsort(-scores)
        feats_sorted = np.asarray(all_candidates['feats'])[best_indices]

        selected_indices = []
        limits = list(self.max_splits_each_feature)
        if len(limits) > 0:
            feat_counts = [0] * len(limits)
            remaining_total = sum(limits)
            for idx, feat in zip(best_indices, feats_sorted):
                if feat_counts[feat] < limits[feat]:
                    feat_counts[feat] += 1
                    selected_indices.append(idx)
                    remaining_total -= 1
                    if remaining_total == 0:
                        break

        else: selected_indices = best_indices[:self.max_splits]

        best_cand = SplitType.create_empty_split_directory()
        for key in KEYS:
            source = all_candidates[key]
            if isinstance(source, np.ndarray):
                best_cand[key] = source[selected_indices].tolist()
            else:
                best_cand[key] = [source[i] for i in selected_indices]

        #best_cand['scores'] = scores[selected_indices]
        print(best_cand['feats'])
        print(self.max_splits_each_feature)
        self.splits_dic = best_cand



    def score_splits(self, splits):
        vals = np.unique(splits)
        is_0 = (splits == vals[0])

        def gini():
            prob_0 = np.mean(is_0, axis=1)
            prob_1 = 1.0 - prob_0
            return 1 - np.abs(prob_0 - prob_1)

        gini_scores = gini()

        def interval_score():
            scores = np.array([self.interval_error(np.where(row_mask)[0]) + self.interval_error(np.where(~row_mask)[0])for row_mask in is_0])
            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return 1 - scores / max_s

        inter_scores = interval_score()

        def mse_score():
            scores = np.array([self.mse_error(np.where(row_mask)[0]) + self.mse_error(np.where(~row_mask)[0]) for row_mask in is_0])
            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return 1 - scores / max_s

        mse_scores = mse_score()

        def cluster_score():
            valid_mask = np.array([np.sum(m) >= 2 and np.sum(~m) >= 2 for m in is_0])
            scores = np.full(len(is_0), np.nan)

            for idx in np.where(valid_mask)[0]:
                m = is_0[idx]
                scores[idx] = self.cluster_error(np.where(m)[0]) + self.cluster_error(np.where(~m)[0])

            valid_scores = scores[valid_mask]
            if len(valid_scores) > 0 and np.ptp(valid_scores) > 0:
                scores[valid_mask] = 1.0 - (valid_scores - np.min(valid_scores)) / np.ptp(valid_scores)
            scores[~valid_mask] = 0.0
            return scores

        cluster_scores = cluster_score()

        return 0.5*cluster_scores + 0.5*gini_scores + 0.3*inter_scores + 0.2*mse_scores


    def map_samples_to_splits(self, samples):
        if self.splits_dic is None:
            print('cant map splits, since its none')
            return None
        split_classes = self.splits_dic['types']
        vals = self.splits_dic['values']
        feats = self.splits_dic['feats']

        new_splits = np.zeros(((len(split_classes)), samples.shape[0]))
        for i, split in enumerate(split_classes):
            new_splits[i] = np.array(split.map_samples_to_split(samples, feats[i], vals[i]))

        return new_splits

    def evaluate_sample(self,index, sample):
        split_classes = self.splits_dic['types']
        vals = self.splits_dic['values']
        feats = self.splits_dic['feats']
        return split_classes[index].evaluate_sample(sample,feats[index],vals[index])

    def left_interval(self,index):
        split_classes = self.splits_dic['types']
        vals = self.splits_dic['values']
        feats = self.splits_dic['feats']
        return split_classes[index].left_interval(feats[index],vals[index])

    def right_interval(self,index):
        split_classes = self.splits_dic['types']
        vals = self.splits_dic['values']
        feats = self.splits_dic['feats']
        return split_classes[index].right_interval(feats[index],vals[index])



###############################"
######helpers


def _generate_splits(max_splits_each_feature,features):
    candidates = SplitType.create_empty_split_directory()
    all_thresholds = calc_all_cont_tresholds(max_splits_each_feature,features)
    BiggerEqSplit.generate_splits(candidates,all_thresholds,features)
    EqualSplit.generate_splits(candidates,all_thresholds,features)
    IntervalSplit.generate_splits(candidates,all_thresholds,features)
    return candidates


def calc_all_cont_tresholds(max_splits_each_feature, features):
    tresh_holds = {}
    for i, feature in enumerate(features):
        if len(max_splits_each_feature) == features.shape[0] and max_splits_each_feature[i] == 0:
            tresh_holds[i] = np.array([])
            continue
        # Percentile binning
        percentiles = np.linspace(5, 95, CONFIG.TRESHHOLD_GRAIN)
        p_thresholds = np.unique(np.percentile(feature, percentiles))

        # Density Valleys (KDE local minima)
        valley_thresholds = find_kde_valleys(feature, num_points=feature.size)

        # Clustering Thresholds
        cluster_thresholds = []
        try:
            cluster = DivisiveCluster()
            cluster.max_depth = CONFIG.TRESHHOLD_GRAIN
            cluster.fit(feature)
            for k in range(1, CONFIG.TRESHHOLD_GRAIN):
                intervals = cluster.get_clusters_at_depth(k)
                for interval in intervals:
                    cluster_thresholds.extend([interval[0], interval[1]])
        except Exception:
            pass

        tresh_holds[i] = np.unique(np.concatenate((p_thresholds, valley_thresholds, cluster_thresholds)))  # remove dubles
    return tresh_holds






def find_kde_valleys(feature: np.ndarray, num_points: int = 100) -> np.ndarray:
    if len(feature) < 10 or np.all(feature == feature[0]): return np.array([])
    try:
        grid = np.linspace(np.min(feature), np.max(feature), max(100, num_points))
        kde = gaussian_kde(feature, bw_method=0.15)
        density = kde(grid)
        minima_indices = (density[1:-1] < density[:-2]) & (density[1:-1] < density[2:])
        valleys = grid[1:-1][minima_indices]

        if len(valleys) == 0:
            kde_alt = gaussian_kde(feature, bw_method=0.3)
            density_alt = kde_alt(grid)
            minima_indices_alt = (density_alt[1:-1] < density_alt[:-2]) & (density_alt[1:-1] < density_alt[2:])
            valleys = grid[1:-1][minima_indices_alt]
        return valleys
    except Exception:
        return np.array([])