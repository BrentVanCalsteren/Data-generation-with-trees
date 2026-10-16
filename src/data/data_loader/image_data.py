import os
from pathlib import Path
import random
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import CONFIG
import re

valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']


class ImageData:
  def __init__(self, file_path: str | Path, n_max_each_class: Optional[int] = None, seed: Optional[int] = 42):
    self.file_path = Path(file_path)
    self.chunk_size = CONFIG.CHUNK_SIZE
    self.n_max_each_class = n_max_each_class
    self.seed = seed
    self.X: Optional[np.ndarray] = None
    self.Y: Optional[np.ndarray] = None
    self.img_shape: Optional[Tuple[int, ...]] = None
    self.n_channels: int = 1
    self.is_greyscale: bool = False
    self.chunk_files: List[Path] = []
    self.n_chunks: int = 0
    self.class_to_idx: dict = {}
    self.split_if_needed()

  def split_if_needed(self):
    if not self.file_path.exists():
      raise FileNotFoundError(f"directory {self.file_path} doesn't exist")

    parent_dir = self.file_path.parent
    stem = self.file_path.name

    # Check if chunks are already saved on PC
    existing_chunks = [p for p in parent_dir.glob(f'{stem}_chunk_*.csv') if p.is_file()]
    if existing_chunks:
      existing_chunks.sort(key=lambda p: int(re.search(r'_chunk_(\d+)', p.name).group(1)) if re.search(r'_chunk_(\d+)', p.name) else 0)
      self.chunk_files = existing_chunks
      self.n_chunks = len(existing_chunks)
      print(f"Found {self.n_chunks} pre-existing chunks for '{stem}' in {parent_dir}, skipping chunk creation")
      return self.chunk_files

    class_folders = [f for f in self.file_path.iterdir() if f.is_dir()]
    class_folders.sort()
    if not class_folders: raise ValueError(f'No classes found at: {self.file_path}')

    self.class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}

    all_samples = []
    for folder in class_folders:
      image_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions and f.is_file()]
      image_files.sort()

      if self.n_max_each_class is not None and len(image_files) > self.n_max_each_class:
        rng = random.Random(self.seed)
        image_files = rng.sample(image_files, self.n_max_each_class)

      label_idx = self.class_to_idx[folder.name]
      for img_file in image_files: all_samples.append((str(img_file), label_idx))

    total_samples = len(all_samples)
    if total_samples == 0: raise ValueError(f'no images found in {self.file_path}')

    parent_dir = self.file_path.parent
    stem = self.file_path.name

    if total_samples <= self.chunk_size:
      chunk_path = parent_dir / f'{stem}_chunk_1.csv'
      pd.DataFrame(all_samples, columns=['image_path', 'label']).to_csv(chunk_path, index=False)
      self.chunk_files = [chunk_path]
      self.n_chunks = 1
      return self.chunk_files

    rng = random.Random(self.seed)
    rng.shuffle(all_samples)
    self.chunk_files = []
    self.n_chunks = int(np.ceil(total_samples / self.chunk_size))

    for i in range(self.n_chunks):
      chunk_samples = all_samples[i * self.chunk_size: (i + 1) * self.chunk_size]
      chunk_path = parent_dir / f'{stem}_chunk_{i + 1}.csv'
      pd.DataFrame(chunk_samples, columns=['image_path', 'label']).to_csv(chunk_path, index=False)
      self.chunk_files.append(chunk_path)

    print(f'Dataset contained {total_samples} images. Created {self.n_chunks} creating chunk at: {parent_dir}')
    return self.chunk_files

  def load_chunk(self, chunk_num: int = 0, label_at_front=False, balanced=True):
    chunk_loaction = self.chunk_files[chunk_num]

    try:
      df_chunk = pd.read_csv(chunk_loaction)
      features = []
      labels = []

      for _, row in df_chunk.iterrows():
        img_path = Path(row['image_path'])
        label_id = int(row['label'])

        try:
          img = Image.open(img_path)
          if self.img_shape is None:
            self.is_greyscale = img.mode in ('L', '1')
            img = (img.convert('L') if self.is_greyscale else img.convert('RGB'))
            img_array = np.asarray(img, dtype=np.float32)

            self.img_shape = img_array.shape
            self.n_channels = (1 if self.is_greyscale else (3 if len(self.img_shape) == 3 else 1))
          else:
            img = (img.convert('L') if self.is_greyscale else img.convert('RGB'))
            target_size = (self.img_shape[1], self.img_shape[0])  # (w, h)
            if img.size != target_size:
              img = img.resize(target_size)
            img_array = np.asarray(img, dtype=np.float32)

          features.append(img_array)
          labels.append(label_id)

        except Exception as e: print(f'failed to load image {img_path}: {e}')

      if not features: raise ValueError(f'No valid images loaded {chunk_loaction}')
      self.X = np.array(features)
      self.Y = np.array(labels).reshape(-1, 1)

      print(f'loaded chunk {chunk_num+1}/{self.n_chunks} ({chunk_loaction.name}): {self.X.shape[0]} samples loaded')
      return self.get_samples(balanced)

    except Exception as e: raise RuntimeError(f'error loading chunk from {chunk_loaction}: {e}')

  def image_arr_to_tabular(self) -> np.ndarray:
    if self.X is None: raise ValueError("Data not loaded. Call 'loading_chunk()' first.")
    return self.X.reshape(self.X.shape[0], -1)

  def tabular_to_image_arr(self, tabular: np.ndarray) -> np.ndarray:
    if self.img_shape is None: raise ValueError('Original image shape is unknown. Load a chunk first.')
    return tabular.reshape((tabular.shape[0], *self.img_shape))

  def save_output_to_folder(
      self,
      tabular_data: np.ndarray,
      labels: np.ndarray,
      folder_name='output',
      filename='generated_image',
  ):
    im_data = self.tabular_to_image_arr(tabular_data)
    os.makedirs(folder_name, exist_ok=True)

    for i, label in enumerate(labels):
      img_uint8 = np.clip(im_data[i], 0, 255).astype(np.uint8)
      img_pil = Image.fromarray(img_uint8)
      lbl_val = label[0] if isinstance(label, (np.ndarray, list)) else label
      file_path = os.path.join(
          folder_name, f'{filename}_{lbl_val}.png'
      )
      img_pil.save(file_path)
      print(f'Success! Image saved to: {file_path}')

  def get_samples(self, balanced=False):
      if balanced:
        return balance_samples(self.image_arr_to_tabular(), self.Y)
      return self.image_arr_to_tabular(), self.Y

def balance_samples(samples: np.ndarray, labels: np.ndarray):
    if labels is None or labels.size == 0: return samples, labels

    labels = labels.flatten()
    uniques, counts = np.unique(labels, return_counts=True)
    if len(uniques) <= 1: return samples, labels

    max_count = np.max(counts)
    samples_bals = []
    labels_bals = []
    for cls in uniques:
      ids = np.where(labels == cls)[0]
      resampled_ids = np.random.choice(ids, size=max_count, replace=True)
      samples_bals.append(samples[resampled_ids])
      labels_bals.append(labels[resampled_ids])

    s_balanced = np.vstack(samples_bals)
    l_balanced = np.concatenate(labels_bals).reshape(-1, 1)
    perm = np.random.permutation(len(l_balanced))
    return s_balanced[perm], l_balanced[perm]