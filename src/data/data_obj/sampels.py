import gc

import numpy as np
import CONFIG
from sklearn.model_selection import train_test_split

from data.data_loader.dataLoader import load_dataloader_by_name
from data.data_obj.chunk_info import GlobalChunkInfo
from data.data_obj.feature_history import FeatureHistory
from collections import Counter

class Samples:
    def __init__(self, dataset: str = 'iris', data_type='tabular', labels_at_front=False):
        self.samples = np.array([])
        self.labels = np.array([])
        self.samples_test = np.array([])
        self.labels_test = np.array([])

        self.loader = load_dataloader_by_name(dataset_name=dataset, data_type=data_type)
        self.labels_at_front = labels_at_front
        self.active_chunk: int = None


    def load_new_dataset(self, dataset='iris', data_type='tabular'):
        self.loader = load_dataloader_by_name(dataset_name=dataset, data_type=data_type)


    def load_chunk(self, chunk_id, split_test=0.0):
        if self.active_chunk == chunk_id: return

        self.active_chunk = chunk_id
        samples, lables = self.loader.load_chunk(chunk_id, self.labels_at_front)
        samples = convert_samples_to_num(samples)
        samples_train, samples_test, labels_train, labels_test = train_test(samples=samples, labels=lables, test_size=split_test)
        self.samples = self.pre_process_samples(samples=samples_train)
        self.samples_test = self.pre_process_samples(samples=samples_test)
        self.labels = labels_train
        self.labels_test = labels_test


    def get_best_matching_labels(self, gen_samples: np.ndarray,chunk_id=0,k= 1):
        self.load_chunk(chunk_id)
        labels_flat = np.asarray(self.labels).flatten()
        real_samples = self.samples
        weights = CONFIG.GLOBAL_CHUNK_INFO.feature_importance
        gen_labels = []
        for i, sample in enumerate(gen_samples):
            diff = np.abs(real_samples - sample)
            if weights is not None: diff = diff * weights
            distances = np.sum(diff, axis=1)
            best_ids = np.argpartition(distances, k)[:k]
            best_ids = best_ids[np.argsort(distances[best_ids])]
            possible_labels = labels_flat[best_ids]
            most_common_label = Counter(possible_labels).most_common(1)[0][0]
            gen_labels.append(most_common_label)
        return np.array(gen_labels)

    def pre_process_samples(self, samples):
        if np.array(samples).shape[0] == 0: return samples
        return CONFIG.GLOBAL_CHUNK_INFO.global_preprocessor.preprocess(samples)


    def reverse_process_samples(self, samples):
        if np.array(samples).shape[0] == 0: return samples
        return CONFIG.GLOBAL_CHUNK_INFO.global_preprocessor.reverse_process(samples)

    def get_samples(self):
        return self.samples

    def save_output(self,samples, llables=None, output_name='output'):
        samples = self.reverse_process_samples(samples)
        labels = self.labels
        if llables is not None:
            labels = np.array(llables)
        self.loader.save_output_to_folder(samples,labels, filename=output_name)


#################
#other functions
#################

def train_test(samples, labels, test_size=0.2):
    if test_size == 0.0: return samples, samples[:0], labels, labels[:0]
    if test_size == 1.0: return samples[:0], samples, labels[:0], labels
    return train_test_split(samples, labels, test_size=test_size, random_state=42)

def convert_samples_to_num(samples):
    num_feat = []
    featurs = samples.T
    for i, feat in enumerate(featurs):
        num_feat.append(make_num(feat))
    return np.array(num_feat).T.astype(float)


def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data]).astype(float)
        # maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr