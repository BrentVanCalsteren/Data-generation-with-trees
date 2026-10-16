import random
from typing import List

import numpy as np
from src.samplers.sampler import Sampler
import src.trees.predictors.helpers.interval as INTERVAL

class SingleGaussian1DSampler(Sampler):
    mean = None
    var = None
    log_norm_const = None
    best_relative_score = None

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.mean = points.mean()
        self.var = points.var()

    def score_feature(self, feature: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var)
        if std < 1e-10:
            std = 1e-10
        #this is an aproximation for calculating it faster
        z_score = (feature - self.mean) / std
        score = np.exp(-0.5 * (z_score ** 2))
        return score

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1):
        std = np.sqrt(self.var)
        return np.random.normal(loc=self.mean, scale=std, size=n_samples)

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]

    def sample_with_confidence(self, n_samples: int = 1, conf_thresh: float = 0.8, max_attempts: int = 100) -> np.ndarray:
        accepted_samples = []
        attempts = 0
        batch_size = max(n_samples * 4, 50)
        while len(accepted_samples) < n_samples and attempts < max_attempts:
            candidates = self.sample(n_samples=batch_size)
            scores = self.score_feature(candidates)
            valid = candidates[scores >= conf_thresh]
            accepted_samples.extend(valid)
            attempts += 1
        if len(accepted_samples) >= n_samples:
            return np.array(accepted_samples[:n_samples])
        return self.sorted_samples(n=n_samples) #if cant find candidates


    @classmethod
    def sample_from_interval_list_single_feature(cls, interval_list, n=1):
        samples = []
        if len(interval_list) == 0: return np.array([])
        for i in range(n):
            interval = interval_list[random.randint(0,len(interval_list)-1)] #select random interval
            np_list = INTERVAL.return_interval_as_list(interval)
            mean = np_list.mean()
            std = np.sqrt(np_list.var())
            samples.append(np.random.uniform(low=np_list[0],high=np_list[1],size=1))
        return np.array(samples).flatten()