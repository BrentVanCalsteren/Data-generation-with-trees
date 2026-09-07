import numpy as np
from typing import List, Optional, Union

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory, get_feature_weights
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from src.usePydl.predictors.predictor import Predictor
from src.usePydl.predictors.helpers.tree import Tree
import src.usePydl.predictors.helpers.interval as INTERVAL
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.evalData.discriminator import NNDiscriminator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class LeafCoherenceBoostingPredictor:
    def __init__(self, sample_obj, n_trees: int = 5, agrasivness: float = 1.0):
        self.sample_obj = sample_obj
        self.n_trees = n_trees
        self.agrasivness = agrasivness
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.n_chunks = self.chunk_info.n_chunks

        self.trees: List[Tree] = []
        self.tree_weights: List[float] = []
        self.build_ensemble()

    def build_ensemble(self):
        print(f"Starting  LeafCoherenceBoosting ({self.n_trees} trees ===")
        for chunk in range(self.n_chunks):
            self.sample_obj.load_chunk(chunk)
            samples = self.sample_obj.samples
            n_samples = samples.shape[0]
            sample_weights = np.ones(n_samples, dtype=np.float64) / n_samples  # init uniform
            for m in range(self.n_trees//self.n_chunks):
                print(f"training tree based on sample weights: ({sample_weights}")
                sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)  # sample based on weight
                boost_samples = np.unique(samples[sampled_indices],return_counts=False, axis=0)

                feat_hist = FeatureHistory(samples=boost_samples) #create splits
                feat_weights = get_feature_weights(mode='random')
                feat_hist.creat_splits(weights_each_feature=feat_weights)


                p = Predictor(sub_samples=boost_samples,n_total_samples=n_samples,splits=feat_hist.get_splits_array(), weights=feat_weights)
                feat_hist.tree = p.tree
                if feat_hist.get_splits_array().shape[0] < 2:
                    print('BREAKING EARLY, CANT GENERATE ENOUGH USEFULL SPLITS')
                    break
                complete_tree = feat_hist.get_complete_tree_memory_efficient()
                # 5. Compute per-sample residual error on the training data
                leafs = complete_tree.get_leafs()
                for leaf in leafs:
                    leaf_error = leaf["error"]
                    sample_ids = leaf["value"]["sample_ids"]
                    orig_ids = list(set(sampled_indices[sample_ids]))
                    leaf_weights = sample_weights[orig_ids]
                    sample_weights[orig_ids] = leaf_error * leaf_weights


                self.trees.append(complete_tree)
                self.tree_weights.append(1) #don't know why i should give a weight to tree
                sample_weights[sample_weights < 1e-8] = 0
                sample_weights /= np.sum(sample_weights)


class GANBoostingPredictor:
    #updating the tree with the help of a discriminator
    def __init__(self, sample_obj, n_trees: int = 5, agrasivness: float = 1.0):
        self.sample_obj = sample_obj
        self.n_trees = n_trees
        self.agrasivness = agrasivness
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.n_chunks = self.chunk_info.n_chunks

        self.trees: List[Tree] = []
        self.tree_weights: List[float] = []
        self.build_ensemble()


    def build_ensemble(self):
        print(f"Starting  GANBoosting ({self.n_trees} trees ===")
        for chunk in range(self.n_chunks):
            self.sample_obj.load_chunk(chunk)
            samples = self.sample_obj.samples
            n_samples = samples.shape[0]
            sample_weights = np.ones(n_samples, dtype=np.float64) / n_samples  # init uniform
            for m in range(self.n_trees // self.n_chunks):
                print(f"training tree based on sample weights: ({sample_weights}")
                sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)  # sample based on weight
                boost_samples = np.unique(samples[sampled_indices],return_counts=False, axis=0)

                feat_hist = FeatureHistory(samples=boost_samples)  # create splits
                feat_weights = None #get_feature_weights(mode='random')
                feat_hist.creat_splits(weights_each_feature=feat_weights)

                p = Predictor(sub_samples=boost_samples,n_total_samples=n_samples,splits=feat_hist.get_splits_array(), weights=feat_weights)
                feat_hist.tree = p.tree
                if feat_hist.get_splits_array().shape[0] < 2:
                    print('BREAKING EARLY, CANT GENERATE ENOUGH USEFULL SPLITS')
                    break
                complete_tree = feat_hist.get_complete_tree_memory_efficient()
                self.trees.append(complete_tree)

                n_gen = min(n_samples, 500)
                fake_samples = gen_new_data_guided(predictor=self,n=n_gen, reference_samples=samples)

                disc = self._train_discriminator(dtype='nn',X_real=samples,X_fake=fake_samples)

                def _predict_prob_real(discriminator, X: np.ndarray) -> np.ndarray:
                    preds = discriminator.predict(X)
                    return preds.flatten().astype(float)

                # 5. Evaluate P(real | x) on the original real dataset
                probs_real = _predict_prob_real(disc, samples)
                probs_real = np.clip(probs_real, 0.05, 0.95)

                # 6. Density Ratio log-odds update
                # Points where P(real) is close to 1 are under-represented by the generator
                log_odds = np.log(probs_real / (1.0 - probs_real))
                clipped_odds = np.clip(log_odds, -2.5, 2.5)

                # 7. Update sample weights
                sample_weights = sample_weights * np.exp(self.agrasivness * clipped_odds)
                sample_weights[sample_weights < 1e-10] = 0
                sample_weights /= np.sum(sample_weights)


    def _train_discriminator(self,X_real: np.ndarray, X_fake: np.ndarray, dtype='nn'):
        if dtype == "nn":
            nn_disc = NNDiscriminator(epochs=25, batch_size=64)
            nn_disc.fit(X_real, X_fake)
            return nn_disc

        n_real = X_real.shape[0]
        n_fake = X_fake.shape[0]
        X_disc = np.vstack([X_real, X_fake])
        y_disc = np.hstack([np.ones(n_real), np.zeros(n_fake)])

        if dtype == "logistic": clf = LogisticRegression(max_iter=200, random_state=42)
        else: clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        clf.fit(X_disc, y_disc)
        return clf


def gen_new_data_guided(predictor, n=100, reference_samples: np.ndarray = None, conf=0.8):

    if reference_samples is None:
        predictor.sample_obj.load_chunk(0)
        reference_samples = predictor.sample_obj.samples

    n_ref = reference_samples.shape[0]
    indices_ref = np.random.default_rng().choice(n_ref, size=n, replace=True)
    gen_samples = np.zeros((n, len(predictor.chunk_info.processed_feat_min)))
    for gen_id, ref_id in enumerate(indices_ref):
        intervals = []
        # loop over all trees
        for t in predictor.trees:
            p = t.let_sample_treverse_tree(reference_samples[ref_id])
            split_infos = p['splits']
            directions = p['directions']
            for j, d in enumerate(directions):
                split_info = split_infos[j]
                idx = split_info['id']
                split_obj = split_info['split_obj']
                if d == 'L':
                    intervals.append(split_obj.left_interval(idx))
                else:
                    intervals.append(split_obj.right_interval(idx))
        # sample values from all intervals
        for feat in range(reference_samples.shape[1]):
            combined_inter = [
                (True, predictor.chunk_info.processed_feat_min[feat], predictor.chunk_info.processed_feat_max[feat], True)]
            for inter in intervals:
                inter = inter.get(feat, None)
                if inter is not None:
                    combined_inter = INTERVAL.intersect_union_interval_lists(combined_inter, inter)

            for inter in combined_inter:
                if INTERVAL.contains_value(inter, reference_samples[ref_id, feat]): gen_samples[gen_id, feat] = \
                SingleGaussian1DSampler.sample_from_interval(interval=inter)[0]
    return gen_samples