import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import CONFIG

MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']


class TabularData:

    def __init__(self, file_path: str | Path, seed: Optional[int] = 42):
        self.file_path = Path(file_path)
        self.chunk_size = CONFIG.CHUNK_SIZE
        self.seed = seed
        self.complete_X: Optional[np.ndarray] = None
        self.missing_X: Optional[np.ndarray] = None
        self.chunk_files: List[Path] = []
        self.n_chunks: int = 0
        self.split_if_needed()

    def split_if_needed(self) -> List[Path]:
        if not self.file_path.exists():
          raise FileNotFoundError(f'File not found: {self.file_path}')

        parent_dir = self.file_path.parent
        stem = self.file_path.stem
        ext = self.file_path.suffix or '.csv'

        # Check if chunks are already saved on PC
        existing_chunks = [p for p in parent_dir.glob(f'{stem}_chunk_*{ext}') if p.is_file()]
        if existing_chunks:
          import re
          existing_chunks.sort(
              key=lambda p: int(re.search(r'_chunk_(\d+)', p.name).group(1)) if re.search(r'_chunk_(\d+)',
                                                                                          p.name) else 0)
          self.chunk_files = existing_chunks
          self.n_chunks = len(existing_chunks)
          print(f"Found {self.n_chunks} pre-existing chunks for '{stem}' in {parent_dir}. Skipping chunk creation.")
          return self.chunk_files

        df = pd.read_csv(
          self.file_path,
          sep=None,
          header=0,
          encoding='utf-8',
          engine='python',
          on_bad_lines='skip',
        )

        total_samples = len(df)
        if total_samples <= self.chunk_size:
          self.chunk_files = [self.file_path]
          self.n_chunks = 1
          return self.chunk_files

        # Randomly shuffle rows for unique chunking
        df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        self.chunk_files = []
        self.n_chunks = int(np.ceil(total_samples / self.chunk_size))

        for i in range(self.n_chunks):
          chunk_df = df_shuffled.iloc[i * self.chunk_size: (i + 1) * self.chunk_size]
          chunk_path = parent_dir / f'{stem}_chunk_{i + 1}{ext}'
          chunk_df.to_csv(chunk_path, index=False)
          self.chunk_files.append(chunk_path)

        print(f'Dataset: {total_samples} samples Split into {self.n_chunks} stored in: {parent_dir}')
        return self.chunk_files

    def load_chunk(self, chunk_num: int = 0,label_at_front=False, balanced=True):
        chunk_loaction = self.chunk_files[chunk_num]
        df = pd.read_csv(
          chunk_loaction,
          sep=None,
          header=0,
          encoding='utf-8',
          engine='python',
          on_bad_lines='skip',
        )
        raw_samples = df.to_numpy()
        missing_mask = _get_missing_mask(raw_samples)
        self.complete_X = raw_samples[~missing_mask]
        self.missing_X = raw_samples[missing_mask]

        print(
          f'Loaded chunk {chunk_num}/{self.n_chunks} ({chunk_loaction.name}): '
          f'complete {self.complete_X.shape[0]}, missing'
          f' {self.missing_X.shape[0]}'
        )

        return self.get_samples(label_at_front,balanced)


    def save_output_to_folder(self, tabular_data: np.ndarray, labels: np.ndarray, folder_name='output', filename='output.csv',):
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, filename)

        if labels is not None:
          data_to_save = np.hstack([tabular_data, labels.reshape(-1, 1)])
        else:
          data_to_save = tabular_data

        np.savetxt(file_path, data_to_save, delimiter=',', fmt='%s')
        print(f'Successfully saved data to: {file_path}')

    def get_samples(self, label_at_front = False, balanced=True):
        if self.complete_X is None:
          raise ValueError("Call loading_chunk() first")
        complete_X = convert_samples_to_num(self.complete_X)
        if label_at_front:
            samples = complete_X[:, 1:]
            labels = complete_X[:, 0].flatten()
        else:
            samples = complete_X[:, :-1]
            labels = complete_X[:, -1].flatten()
        if balanced:
            return balance_samples(samples, labels)
        return samples, labels


def _get_missing_mask(data: np.ndarray) -> np.ndarray:
    mask = np.zeros(data.shape[0], dtype=bool)
    for col in range(data.shape[1]):
        col_data = data[:, col].astype(str)
        mask |= np.isin(col_data, MISSING_VAL_STRINGS)
    return mask

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


def balance_samples(samples: np.ndarray, labels: np.ndarray):
    if labels is None or labels.size == 0: return samples, labels
    labels = labels.flatten()
    uniques, counts = np.unique(labels, return_counts=True)
    if len(uniques) <= 1: return samples, labels

    max_count = np.max(counts)
    # If all classes already have the same count, do not resample!
    if np.all(counts == max_count):
        return samples, labels

    samples_bals = []
    labels_bals = []
    for cls in uniques:
        ids = np.where(labels == cls)[0]
        if len(ids) == max_count:
            resampled_ids = ids
        else:
            extra_ids = np.random.choice(ids, size=max_count - len(ids), replace=True) #only add additional copies
            resampled_ids = np.concatenate([ids, extra_ids])
        samples_bals.append(samples[resampled_ids])
        labels_bals.append(labels[resampled_ids])

    s_balanced = np.vstack(samples_bals)
    l_balanced = np.concatenate(labels_bals).reshape(-1, 1)
    perm = np.random.permutation(len(l_balanced))
    return s_balanced[perm], l_balanced[perm]