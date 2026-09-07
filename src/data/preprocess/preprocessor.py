import numpy as np

from src.data.preprocess.scaler import Scaler
from src.data.preprocess.encoder_PCA import PCAEncoder
from src.data.preprocess.encoder_NN import NNencoder
import CONFIG

class Processor:
    def __init__(self, process_list,samples=None):
        self.processes = []
        for p in process_list:
            if isinstance(p, str): self.add_process(p)
            if isinstance(p, list):
                if samples is not None:
                    if samples.shape[1] <= CONFIG.REDUCE_FEAT: continue
                self.add_process(p[0],p[1])


    def add_process(self, process_str: str, extra_info: str = None):
        if process_str == "scale": self.processes.append(Scaler())
        elif process_str == "rotate_dim": self.processes.append(PCAEncoder())
        elif process_str == "reduce_feat":
            if extra_info == "pca":
                self.processes.append(PCAEncoder(output_dim=CONFIG.REDUCE_FEAT))
            elif extra_info == "nn":
                self.processes.append(NNencoder(output_dim=CONFIG.REDUCE_FEAT))

    def partial_fit_process(self,samples : np.ndarray, process_id = 0):
        try:
            self.processes[process_id].partial_fit(samples)
        except:
            print('process id out of range')

    def preprocess(self, samples : np.ndarray):
        for p in self.processes:
            samples = p.transform(samples)

        return samples

    def reverse_process(self,samples : np.ndarray):
        for p in reversed(self.processes):
            samples = p.inverse_transform(samples)

        return samples

    def get_feature_importance(self):
        importances = None
        importances_nn = []
        importances_pca = []
        len_nn = 0
        len_pca = 0
        for i in range(len(self.processes)-1,0,-1):
            p = self.processes[i]
            if isinstance(p, NNencoder):
                if len_nn == 0:
                    importances_nn = p.get_explained_variance_ratio()
                    len_nn = len(importances_nn)
            if isinstance(p, PCAEncoder):
                if len_pca == 0:
                    importances_pca = p.get_explained_variance_ratio()
                    len_pca = len(importances_pca)
        if  min(len_nn, len_pca)> 0:
            l = min(len_nn, len_pca)
            importances = importances_nn[:l]*importances_pca[:l]
        elif len_nn > 0:
            importances = importances_nn
        elif len_pca > 0:
            importances = importances_pca
        return importances










