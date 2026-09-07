from abc import ABC, abstractmethod
import numpy as np


class SplitType(ABC):


    @classmethod
    def generate_splits(cls,candidates,tresholds, features: np.ndarray):
        pass

    @classmethod
    def create_empty_split_directory(cls):
        return {'types' : [],
                'values' : [],
                'feats' : [],
                'splits' : [],
                'scores': [],}

    @classmethod
    def add_split(cls, candi, feats, splits: np.ndarray, vals):
        found_splits = candi['splits']
        inverse_split = ~splits
        already_exists = any( np.array_equal(splits, s) or np.array_equal(inverse_split, s) for s in found_splits)

        if not already_exists:
            candi['types'].append(cls)
            candi['values'].append(vals)
            candi['splits'].append(splits)
            candi['feats'].append(feats)

    @classmethod
    def add_splits(cls, candi, feats: int, splits_matrix: np.ndarray, vals_list):
        if '_seen_hashes' not in candi:
            candi['_seen_hashes'] = set()
            for s in candi['splits']:
                candi['_seen_hashes'].add(s.tobytes())
                candi['_seen_hashes'].add((~s).tobytes())

        seen = candi['_seen_hashes']

        new_types = []
        new_vals = []
        new_splits = []
        new_feats = []

        for split, val in zip(splits_matrix, vals_list):
            split_bytes = split.tobytes()
            inv_bytes = (~split).tobytes()

            # Check in O(1) time
            if split_bytes not in seen and inv_bytes not in seen:
                seen.add(split_bytes)
                seen.add(inv_bytes)

                new_types.append(cls)
                new_vals.append(val)
                new_splits.append(split)
                new_feats.append(feats)

        # 2. Bulk append to lists
        candi['types']+=new_types
        candi['values']+=new_vals
        candi['splits']+=new_splits
        candi['feats']+=new_feats




"""


class LinearCombinationSplit(SplitType):
    def __init__(self, feat_ids: List[int], weights: List[float], threshold: float):
        super().__init__(feat_ids)
        self.weights = [float(w) for w in weights]
        self.threshold = float(threshold)

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples)
        if samples.ndim == 1:
            z = sum(w * (samples[f] if samples.size > f else 0.0) for f, w in zip(self.feat_ids, self.weights))
            return np.array([z >= self.threshold], dtype=bool)
        z = np.zeros(samples.shape[0], dtype=float)
        for f, w in zip(self.feat_ids, self.weights):
            z += w * samples[:, f]
        return z >= self.threshold

    def add_left_interval(self, intervals: Intervals):
        #not implemented
        pass

    def add_right_interval(self, intervals: Intervals):
        #not implemented
        pass

    def to_string(self) -> str:
        f1, f2 = self.feat_ids[0], (self.feat_ids[1] if len(self.feat_ids) > 1 else self.feat_ids[0])
        w1, w2 = self.weights[0], (self.weights[1] if len(self.weights) > 1 else 0.0)
        return f"lincomp_{f1}_{f2}_{w1}_{w2}_>=_{self.threshold}"

    @classmethod
    def generate_nf_splits(cls,candidates,thresholds, features: np.ndarray):
        pass
        
"""

