import numpy as np
from pyexpat import features

import CONFIG
from src.data.preprocess.preprocessor import Processor

class GlobalChunkInfo:
    def __init__(self, loader, labels_at_front=False):
        self.loader = loader
        self.n_chunks = loader.n_chunks
        self.labels_at_front = labels_at_front
        self.global_preprocessor: Processor = None

        self.feats_min = None
        self.feats_max = None
        self.processed_feat_min = None
        self.processed_feat_max = None
        self.feature_importance = None

        self.featureTypes = None
        self.discrete_values = None
        self.total_number_samples = 0

        self.corr_matrix = None
        self.class_corrs = {}

        self.init_all_parameters()
        self.n_features = len(self.processed_feat_min)
        self.start_intervals = []
        for i in range(len(self.processed_feat_min)):
            self.start_intervals.append([(True, self.processed_feat_min[i], self.processed_feat_max[i], True)])

    def init_all_parameters(self):
        preproces_done = []
        for i in range(self.n_chunks):
            chunk_samples, labels = self.loader.load_chunk(i, self.labels_at_front)
            chunk_samples = convert_samples_to_num(chunk_samples)

            if self.global_preprocessor is None:
                self.global_preprocessor = Processor(CONFIG.PREPROCESS_LIST, chunk_samples)

            self.total_number_samples += chunk_samples.shape[0]
            if self.feats_min is None: self.feats_min= np.min(chunk_samples, axis=0)
            else: self.feats_min = np.minimum(self.feats_min, np.min(chunk_samples, axis=0))
            if self.feats_max is None: self.feats_max = np.max(chunk_samples, axis=0)
            else: self.feats_max = np.maximum(self.feats_max, np.max(chunk_samples, axis=0))
            self.update_discrete(chunk_samples)

        #init preprocessing
        for i in range(len(self.global_preprocessor.processes)):
            for j in range(self.n_chunks):
                chunk_samples, _ = self.loader.load_chunk(j, self.labels_at_front)
                chunk_samples = convert_samples_to_num(chunk_samples)
                for k in range(i): chunk_samples = self.global_preprocessor.processes[k].transform(chunk_samples)
                self.global_preprocessor.processes[i].partial_fit(chunk_samples)



        try:
            for i in range(self.loader.n_chunks):
                chunk_samples, _ = self.loader.load_chunk(i, self.labels_at_front)
                chunk_samples = convert_samples_to_num(chunk_samples)
                p_samples = self.global_preprocessor.preprocess(chunk_samples)
                if self.processed_feat_min is None:
                    self.processed_feat_min = np.min(p_samples, axis=0)
                    self.processed_feat_max = np.max(p_samples, axis=0)
                else:
                    self.processed_feat_min = np.minimum(self.processed_feat_min, np.min(p_samples, axis=0))
                    self.processed_feat_max = np.maximum(self.processed_feat_max, np.max(p_samples, axis=0))

            self.feature_importance = self.global_preprocessor.get_feature_importance()

            print(f'feature_importance: {self.feature_importance}')
        except:
            if self.processed_feat_min is None:
                self.processed_feat_min = self.feats_min
                self.processed_feat_max = self.feats_max
            print('tried getting feature importance, failed')


    def update_discrete(self, samples,discrete_percentile=5):
        if self.discrete_values is None: self.discrete_values = [[] for _ in range(samples.shape[1])]
        if self.featureTypes is None: self.featureTypes = [[] for _ in range(samples.shape[1])]
        for i in range(samples.shape[1]):
            u_vals = list(np.unique(samples[:,i].T))
            u_len = len(u_vals)
            if self.total_number_samples > 0 and u_len <= (self.total_number_samples * discrete_percentile // 100):
                self.discrete_values[i] = list(set(u_vals+self.discrete_values[i]))
                self.featureTypes[i] = 'discrete'
            else:
                self.featureTypes[i] = 'continuous'
                self.discrete_values[i] = []


def convert_samples_to_num(samples):
        num_feat = []
        featurs = samples.T
        for i,feat in enumerate(featurs):
            num_feat.append(make_num(feat))
        return np.array(num_feat).T.astype(float)


def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data]).astype(float)
        #maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr