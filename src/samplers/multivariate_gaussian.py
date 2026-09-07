from typing import Dict, List

import numpy as np
from scipy.stats import multivariate_normal
from src.samplers.sampler import Sampler

class MultivariateGaussianSampler(Sampler):
    def __init__(self):
        self.mean = None
        self.cov = None
        self.dist = None

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        self.mean = np.mean(points, axis=0)
        # Add a small regularizer to the diagonal to prevent singular matrix
        if points.shape[0] > 1:
            self.cov = np.cov(points, rowvar=False) + np.eye(points.shape[1]) * 1e-6
        else:
            self.cov = np.eye(points.shape[1]) * 1e-6 #(Identity Matrix)
        self.dist = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)


    def fit_from_intervals(self, min_vals: np.ndarray, max_vals: np.ndarray, corr_matrix = None):
        min_vals = np.asarray(min_vals, dtype=np.float64)
        max_vals = np.asarray(max_vals, dtype=np.float64)

        self.mean = (min_vals + max_vals) / 2.0
        spans = np.maximum(1e-6, max_vals - min_vals)
        sigmas = spans / 4.0  # 4-sigma spans 95% of the Gaussian distribution

        d = len(min_vals)
        if corr_matrix is not None and corr_matrix.shape == (d, d):
            cov = np.outer(sigmas, sigmas) * corr_matrix
            # Ensure positive semi-definiteness
            self.cov = cov + np.eye(d) * 1e-6
        else:
            self.cov = np.diag(sigmas ** 2) + np.eye(d) * 1e-6
        self.dist = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)

    def sample_with_confidence(self, n_samples = 1, conf_thresh = 0.8, max_attempts = 100):
        accepted_samples = []
        attempts = 0
        batch_size = max(n_samples * 4, 50)
        while len(accepted_samples) < n_samples and attempts < max_attempts:
            candidates = self.sample(n_samples=batch_size)
            scores = self.score_feature(candidates)
            valid = candidates[scores >= conf_thresh]
            if valid.size > 0:
                for row in valid:
                    accepted_samples.append(row)
            attempts += 1
        if len(accepted_samples) >= n_samples:
            return np.array(accepted_samples[:n_samples])
        return self.sorted_samples(n=n_samples)

    def score_feature(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)

        raw_pdf = self.dist.pdf(feature)
        max_pdf = self.dist.pdf(self.dist.mean)
        scaled_score = raw_pdf / max_pdf
        return scaled_score

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    #this takes very long to generate good samples when there are a lot of features
    def sample(self, n_samples=1):
        samples = self.dist.rvs(size=n_samples)
        if n_samples == 1 and samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 1 and self.mean.shape[0] == 1:
            samples = samples.reshape(-1, 1)
        return samples

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]

