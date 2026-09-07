from abc import ABC

import numpy as np

import CONFIG
from splits.split_type import SplitType
from itertools import combinations_with_replacement


#########################################
########### INTERVAL
###################################"


class IntervalSplit(SplitType):

    @classmethod
    def evaluate_sample(cls, sample, feats,vals):
        if  (sample[feats] >= vals[0]) & (sample[feats] <= vals[1]):
            return 'L'
        return 'R'

    @classmethod
    def left_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat:[(True,vals[0], vals[1], True)]}

    @classmethod
    def right_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat:[
            (True,CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[feats], vals[0], False),
            (False,vals[1], CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[feats], True)
        ]}

    @classmethod
    def map_samples_to_split(cls, samples: np.ndarray, feats, vals):
        return (samples.T[feats] >= vals[0]) & (samples.T[feats] <= vals[1])


    @classmethod
    def generate_splits(cls, candidates, thresholds, features: np.ndarray):
        for i, feat in enumerate(features):
            thresh = np.unique(thresholds[i])
            if len(thresh) == 0:
                continue

            pairs = np.array(list(combinations_with_replacement(thresh, 2)))
            lows = pairs[:, 0]
            highs = pairs[:, 1]
            feat_row = feat[None, :]
            splits_matrix = (feat_row >= lows[:, None]) & (feat_row <= highs[:, None])
            cls.add_splits(candidates, i, splits_matrix, pairs)