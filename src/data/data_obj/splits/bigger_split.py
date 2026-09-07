##########################
### BIGGER
################################
import numpy as np

import CONFIG
from splits.split_type import SplitType


class BiggerEqSplit(SplitType):
    @classmethod
    def evaluate_sample(cls, sample, feats,vals):
        if sample[feats] >= vals:
            return 'L'
        return 'R'

    @classmethod
    def left_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat: [(True,vals, CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[feats], True)]}

    @classmethod
    def right_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat:  [(True,CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[feats], vals, False)]}

    @classmethod
    def map_samples_to_split(cls, samples: np.ndarray, feats, vals):
         return samples.T[feats] >= vals

    @classmethod
    def generate_splits(cls, candidates, thresholds, features: np.ndarray):
        for i, feat in enumerate(features):
            feat_row = feat[None, :]
            treshold_colom = thresholds[i][:, None]
            splits_matrix = (feat_row >= treshold_colom)
            cls.add_splits(candidates, i, splits_matrix, thresholds[i])