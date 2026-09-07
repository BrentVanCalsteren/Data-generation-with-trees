import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import CONFIG
from typing import Dict, Any, Optional


def do_all_sanity_checks(real_samples: np.ndarray,gen_samples: np.ndarray,test_samples:np.ndarray = None, real_label:np.ndarray = None,gen_label:np.ndarray = None):
    results = {
        "dim_check": check_dimensions(real_samples, gen_samples),
        "nan_inf_check": check_nan_inf(gen_samples),
        "out_of_bounds": check_out_bounds(real_samples, gen_samples),
        "real_gen_dupes": check_dupes(real_samples, gen_samples),
        "avg_dist_real": avg_dist_vs_real_sampels(real_samples, gen_samples),
        "train_vs_test_dist": closeness_train_vs_test(real_samples, gen_samples, test_samples),
        "label_info": check_label_info_in_data(real_samples, gen_samples,real_label,gen_label)
    }

    print("!SIMPLE DATA SANITY CHECKS!\n"
          + f"-Dimension Checks: {results['dim_check']['result']}\n"
          + f"-No NAN/INF VALS: {results['nan_inf_check']['result']}\n"
          + f"-Num Out of bound VALS (doesn't directly mean it's bad): {results['out_of_bounds']['result']}\n"
          + f"-check dupes: {results['real_gen_dupes']['result']}\n"
          + f"-dist to real: {results['avg_dist_real']['result']}\n"
          + f"-train_vs_test median: {results['train_vs_test_dist']['result']}\n"
          + f"-label info in sample train vs gen: {results['label_info']['result']}\n")

    return results


def check_dimensions(real_samples: np.ndarray, gen_samples: np.ndarray) -> Dict[str, Any]:
    _, f_real = real_samples.shape
    n_gen, f_gen = gen_samples.shape
    passed = 'passed' if (f_real == f_gen and n_gen > 0) else 'failed'
    return {"result": passed}


def check_nan_inf(gen_samples: np.ndarray) -> Dict[str, Any]:
    nans = int(np.isnan(gen_samples).sum())
    infs = int(np.isinf(gen_samples).sum())
    passed = 'passed' if ((nans == 0) and (infs == 0)) else 'failed'
    return {"result": passed}


def check_out_bounds(real_samples: np.ndarray, gen_samples: np.ndarray) -> Dict[str, Any]:
    chunk_info = CONFIG.GLOBAL_CHUNK_INFO
    if chunk_info is not None:
        real_min = chunk_info.processed_feat_min
        real_max = chunk_info.processed_feat_max
    else:
        real_min = np.min(real_samples, axis=0)
        real_max = np.max(real_samples, axis=0)

    gen_min = np.min(gen_samples, axis=0)
    gen_max = np.max(gen_samples, axis=0)

    out_lower = gen_min < real_min
    out_upper = gen_max > real_max
    out_feats = np.logical_or(out_lower, out_upper)

    return {"result": int(np.sum(out_feats))}


def check_dupes(real_samples: np.ndarray, gen_samples: np.ndarray, tol: float = 1e-6) -> Dict[str, Any]:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real_samples)
    dists, _ = nn.kneighbors(gen_samples)

    dupes = (dists <= tol).flatten()
    count = int(np.sum(dupes))

    return {"result": count}


def avg_dist_vs_real_sampels(real_samples: np.ndarray, gen_samples: np.ndarray) -> Dict[str, Any]:
    nn_real = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real_samples)
    _,dist_closest_neighbour = nn_real.kneighbors(gen_samples)
    mean = float(np.mean(dist_closest_neighbour))

    return {"result": mean}


def closeness_train_vs_test(train: np.ndarray, gen_samples: np.ndarray, test: Optional[np.ndarray] = None):
    if test is None: return {"result": "can't compute (test samples is None)"}

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(gen_samples)
    dcr_train, _ = nn.kneighbors(train)
    dcr_test, _ = nn.kneighbors(test)

    med_train = float(np.median(dcr_train))
    med_test = float(np.median(dcr_test))

    ratio = med_test / (med_train + 1e-8)
    return {"result": f"med_train: {med_train}, med_test: {med_test}, ratio: {ratio} (good should be 1, higher-> test close, lower-> train closer)"}


def check_label_info_in_data(real_samples: np.ndarray,gen_samples: np.ndarray, labels_real, labels_gen):
    #compares if the generated samples contain roughly the same info as the real samples
    #check each feature hase same cooreclation to label
    return {"result": f"To Implement"}


