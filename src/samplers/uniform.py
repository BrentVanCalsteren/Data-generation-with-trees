from typing import List
import numpy as np
from src.samplers.sampler import Sampler
import random
import src.trees.predictors.helpers.interval as INTERVAL

class UniformSampler(Sampler):
    min = None
    max = None

    def fit(self, feat):
        self.min = np.min(feat)
        self.max = np.max(feat)

    def score_feature(self, feature) -> np.ndarray: #can be singe point or feature array -> will always return array
        feat_array = np.array(feature)
        inside = ((feat_array >= self.min) & (feat_array <= self.max)).astype(float)
        width = self.max - self.min
        if width == 1: width = 1  - 1e-10
        #the confidence should be higher when the interval is smaller
        return inside * ((1 - width)/10+0.9) #TODO  fix it correctly

    def sample(self, n_samples=1):
        if self.min is None:
            raise RuntimeError("Model must be fitted before sampling.")

        return np.random.uniform(
            low=self.min,
            high=self.max,
            size=n_samples
        )

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        return candidates[np.argsort(-self.score_feature(candidates))]

    def calc_conf_rel_each_sample(self, distributions: List[UniformSampler], samples: np.ndarray):
        features = samples.T
        scores = np.array([dist.score_feature(features[i]) for i, dist in enumerate(distributions)]).T #shape scores (n_samples,n_features)
        return np.mean(scores, axis=1)

    @classmethod
    def sample_from_interval_list_single_feature(cls, interval_list, n=1):
        samples = []
        if len(interval_list) == 0: return np.array([])
        for i in range(n):
            interval = interval_list[random.randint(0,len(interval_list)-1)] #select random interval
            np_list = INTERVAL.return_interval_as_list(interval)
            samples.append(np.random.uniform(low=np_list[0],high=np_list[1],size=1))
        return np.array(samples).flatten()

    @classmethod
    def sample_from_lows_highs_single_feat(cls, lows, highs,weights, n=1):
        samples = []
        low_length = len(lows)
        if low_length != len(highs):
            print('LOW,HIGH MISMATCH')
            return np.array([])
        indices = np.random.choice(low_length, n, p=weights/np.sum(weights))
        for i in indices:
            samples.append(np.random.uniform(low=lows[i], high=highs[i], size=1))
        return np.array(samples).flatten()