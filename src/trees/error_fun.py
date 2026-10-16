from typing import List

import numpy as np
from networkx.generators import trees
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans2

from trees.predictors.helpers.tree import Tree


#continue errors


class CombinedMSEIntervalError:
    def __init__(self, samples: np.ndarray, feature_weights=None, mse_weight: float = 0.6, interval_weight: float = 0.4):
        self.samples = samples
        self.mse_weight = mse_weight
        self.interval_weight = interval_weight
        self.interval_err = IntervalSizesError(samples, feature_weights)
        self.mse_err = MSEError(samples, feature_weights)
        #normalization
        all_ids = np.arange(samples.shape[0], dtype=np.intc)
        self.root_interval = max(1e-5, float(self.interval_err(all_ids)))
        self.root_mse = max(1e-5, float(self.mse_err(all_ids)))
    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        if len(indices) < 2:
            return 10e6
        norm_i = self.interval_err(indices) / self.root_interval
        norm_m = self.mse_err(indices) / self.root_mse
        return (self.interval_weight * norm_i) + (self.mse_weight * norm_m)


class ClusterError:
    def __init__(self, samples, feature_weights=None):
        if feature_weights is not None:
            self.samples = samples * feature_weights
        else:
            self.samples = samples

    def __call__(self, tids):
        # Convert tids without redundant list() overhead
        idx = np.fromiter(tids, dtype=np.int64) if not isinstance(tids, np.ndarray) else tids
        n = len(idx)
        if n < 2:
            return 0.0

        X = self.samples[idx]
        inv_n = 1.0 / n

        # 1. Pass 1: Compute sums and sum of squares per feature
        # Matches the C++ accumulator loop
        sum_f = np.sum(X, axis=0)
        sum_sq_f = np.sum(X ** 2, axis=0)

        # 1-cluster variance per feature: sum_sq - (sum^2 / n)
        var_f = sum_sq_f - (sum_f ** 2 * inv_n)
        var_f = np.maximum(var_f, 0.0)

        w1 = float(np.sum(var_f))
        if w1 < 1e-6:
            return 0.0

        # Find dimension with maximum variance
        max_dim = int(np.argmax(var_f))

        # 2. Pass 2: Split along the mean of the dominant feature
        split_threshold = sum_f[max_dim] * inv_n
        x_dim = X[:, max_dim]

        mask_a = x_dim < split_threshold
        count_a = int(np.count_nonzero(mask_a))
        count_b = n - count_a

        if count_a == 0 or count_b == 0:
            return w1

        sum_split_a = np.sum(x_dim[mask_a])
        mean_a = sum_split_a / count_a
        mean_b = (sum_f[max_dim] - sum_split_a) / count_b

        # 3. Between-cluster variance along primary axis
        between_var = count_a * count_b * inv_n * ((mean_a - mean_b) ** 2)

        # 4. Return W1 + between_var
        return float(w1 + between_var)

    @classmethod
    def copy(cls, error_fun, samples,weights):
        return cls(samples,weights)



#dense bounding boxes
class IntervalSizesError:
    def __init__(self, samples, feature_weights=None):
        if feature_weights is not None:
            self.samples = samples * feature_weights
        else:
            self.samples = samples

    def __call__(self, tids):
        sub_samples = self.samples[list(tids)]
        if sub_samples.shape[0] < 2:
            return 10e7

        max_per_row = sub_samples.max(axis=0)
        min_per_row = sub_samples.min(axis=0)
        diff = max_per_row - min_per_row
        return np.sum(diff)

    @classmethod
    def copy(cls, error_fun, samples,weights):
        return cls(samples,weights)


#finds spherical clusters
class MSEError:
    def __init__(self, samples: np.ndarray, feature_weights=None):
        if feature_weights is not None:
            self.samples = samples * feature_weights
        else:
            self.samples = samples

    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        sub_samples = self.samples[indices]
        if sub_samples.shape[0] < 2:
            return 10e6
        mean = np.mean(sub_samples, axis=0)
        diff_sq = (sub_samples - mean) ** 2
        return np.mean(np.sum(diff_sq, axis=1))

    @classmethod
    def copy(cls, error_fun, samples,weights):
        return cls(samples,weights)

#Mean Absolute Error
class MAEError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples

    def __call__(self, tids):
        sub_samples = self.samples[list(tids)]
        if sub_samples.shape[0] < 2:
            return 10e6
        median = np.median(sub_samples, axis=0)
        return np.mean(np.sum(np.abs(sub_samples - median), axis=1))

    @classmethod
    def copy(cls, error_fun, samples,weights):
        return cls(samples,weights)


#Ensuring tight, strictly bounded clusters
class DiameterError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.good_error = 0.10 * np.sqrt(samples.shape[1])

    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        sub_samples = self.samples[indices]
        if sub_samples.shape[0] < 2:
            return 10e6
        distances = pdist(sub_samples, metric='euclidean')
        if np.max(distances) < self.good_error: return 0
        else: return np.max(distances)

class chainTreeError:
    def __init__(self, error_fun, trees: List, step: int):
        self.samples = error_fun.samples
        self.trees = trees
        self.step = step
        self.error_fun = error_fun
        for tree in self.trees:
            tree.create_intervals_each_paths()

    def __call__(self, tids):
        ids = np.fromiter(tids, dtype=np.int64)
        prev_step = self.step-1
        total_error = 0
        while prev_step >= 0:
            total_error += self.trees[prev_step].miss_class_error(self.samples[ids])
            prev_step -= 1
        return total_error * self.error_fun(tids)

    @classmethod
    def copy(cls, error_fun, samples, weights):
        return cls(error_fun.error_fun.copy(error_fun.error_fun, samples,weights), error_fun.trees, error_fun.step)

##########################
#FAST ERROR FUN
#########################

def min_sup_error(y):
    def error(tids):
        supports = list(tids) # is a list of all counts
        maxindex = np.argmax(supports) #take label that is most promenent
        return sum(supports) - supports[maxindex], maxindex #returns how many missclassified
    return error