import numpy as np

import CONFIG
from splits.split_type import SplitType


######################################
##### Equal
#################################


class EqualSplit(SplitType):
    @classmethod
    def evaluate_sample(cls, sample, feats,vals):
        if sample[feats] == vals:
            return 'L'
        return 'R'

    @classmethod
    def left_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat: [(True,vals, vals, True)]}

    @classmethod
    def right_interval(cls, feats, vals):
        feat = int(feats)
        return {  feat: [(True,CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[feats], vals, False),
                (False,vals, CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[feats],True)]}

    @classmethod
    def map_samples_to_split(cls, samples: np.ndarray, feats, vals):
        return samples.T[feats] == vals

    @classmethod
    def generate_splits(cls,candidates, _, features: np.ndarray):
        for i, feat in enumerate(features):
            uniques, counts = np.unique(feat, return_counts=True)
            for j, count in enumerate(counts):
                if count/features.shape[1] > CONFIG.DESCRETE_PERCENT:
                    k = uniques[j]
                    splits = (feat == k)
                    cls.add_split(candidates, i, splits, k)
