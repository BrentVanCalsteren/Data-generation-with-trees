from pathlib import Path

import CONFIG
from chunk_info import GlobalChunkInfo
from data.data_loader.image_data import ImageData
from data.data_loader.tabular_data import TabularData



def load_dataloader_by_name(dataset_name: str, main_dir: str = 'GAN-aproach-for-boolean-decision-trees-pydl8.5', data_subdir: str = 'datasets', data_type='tabular'):
    file_path = Path().resolve()
    str_path = str(file_path)
    index = str_path.find(main_dir)
    if index == -1: raise ValueError(f"Main directory '{main_dir}' not found in path: {str_path}")
    main_path = Path(str_path[:index + len(main_dir)])
    base_path = main_path / data_subdir
    if data_type == 'tabular':
        for ext in ['.csv', '.data']:
            candidate = base_path / dataset_name / f"{dataset_name}{ext}"
            if candidate.exists():
                print(f"Loading dataset from: {candidate}")
                loader = TabularData(candidate)
    elif data_type == 'image':
            folder = base_path / dataset_name
            loader = ImageData(folder)

    if loader is not None:
        CONFIG.GLOBAL_CHUNK_INFO = GlobalChunkInfo(loader=loader, labels_at_front=False)
        return loader
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {base_path}")


