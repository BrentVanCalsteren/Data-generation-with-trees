import math
import numpy as np
from typing import List

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory, get_feature_weights
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from usePydl.predictors.helpers.tree import Tree
from src.data.data_obj.sampels import Samples
import src.usePydl.predictors.helpers.groups as groups
from src.usePydl.predictors.predictor import Predictor
from src.evalData.discriminator import NNDiscriminator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.usePydl.predictors.data_gen_methods import gen_new_data_guided


class EnsemblePredictor:
    def __init__(self):
        pass

#classic random ensemble approach
class RandomForestPredictor(EnsemblePredictor):
    def __init__(self, sample_obj: Samples, n_trees=0, subsample_ratio: float = 0.7):
        super().__init__()
        self.trees: List[Tree] = []
        self.sample_obj = sample_obj
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.n_chunks = sample_obj.loader.n_chunks
        self.n_features = len(self.chunk_info.processed_feat_min)
        self.features_each_tree = self.n_features
        self.trees_each_chunk = 1
        self.subsample_ratio = subsample_ratio

        n_features_each_tree = int(math.ceil(np.sqrt(self.n_features)))
        if n_features_each_tree > self.n_features:
            self.features_each_tree = int(self.n_features)
            self.trees_each_chunk = 1
        else:
            self.features_each_tree = n_features_each_tree
            self.trees_each_chunk = max(4, int(np.ceil((self.n_features / self.features_each_tree) * 5)))
        self.n_trees = self.n_chunks * self.trees_each_chunk
        print(f"Init Random Forest: {self.n_trees} trees, each {self.features_each_tree} features")
        self.build_ensemble()


    def build_ensemble(self):
        for chunk_id in range(self.n_chunks):
            self.sample_obj.load_chunk(chunk_id)
            chunk_samples = self.sample_obj.samples
            n_total = chunk_samples.shape[0]
            n_sub = int(np.ceil(n_total * self.subsample_ratio))
            sub_indices = np.random.choice(n_total, size=n_sub, replace=False)
            sub_samples = chunk_samples[sub_indices]
            feature_groups = groups.create_total_feat_balanced_groups(n_groups=self.trees_each_chunk,
                                                                       features_each_group=self.features_each_tree,
                                                                       total_features=self.n_features)

            for i, group in enumerate(feature_groups):
                print(f"Tree num: {i}, (chunk:{chunk_id}, feat:{group}) TRAINING")
                feat_hist = FeatureHistory(sub_samples)
                weights = get_feature_weights(mode='random', focus_on=group)
                feat_hist.creat_splits(weights_each_feature=weights)
                __predictor = build_tree_iteratively(feat_hist, weights)
                complete_tree = feat_hist.get_complete_tree_memory_efficient()
                self.trees.append(complete_tree)
                print(f"Tree(chunk:{chunk_id}, feat:{group}) Trained")



#----------------------------

# BOOSTING ENSEMBLE APROACHES

#----------------------------


class LeafCoherenceBoostingPredictor(EnsemblePredictor):
    def __init__(self, sample_obj, n_trees: int = 5, agrasivness: float = 1.0):
        super().__init__()
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
                sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
                unique_orig_ids = np.unique(sampled_indices)
                boost_samples = samples[unique_orig_ids]


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
                errors = [leaf["error"] for leaf in leafs]
                for leaf in leafs:
                    sample_ids = leaf["value"]["sample_ids"]
                    leaf_error = leaf["error"]
                    orig_ids = unique_orig_ids[sample_ids]
                    leaf_weights = sample_weights[orig_ids]
                    sample_weights[orig_ids] = leaf_weights * np.exp(self.agrasivness * (leaf_error / np.max(errors)))


                self.trees.append(complete_tree)
                #self.tree_weights.append(1) #don't know why i should give a weight to tree
                sample_weights[sample_weights < 1e-10] = 0
                sample_weights /= np.sum(sample_weights)


class GANBoostingPredictor(EnsemblePredictor):
    #updating the tree with the help of a discriminator
    def __init__(self, sample_obj, n_trees: int = 5, agrasivness: float = 1.0):
        super().__init__()
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

                probs_real = _predict_prob_real(disc, samples)
                probs_real = np.clip(probs_real, 0.05, 0.95)
                log_odds = np.log(probs_real / (1.0 - probs_real))
                clipped_odds = np.clip(log_odds, -2.5, 2.5)
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



