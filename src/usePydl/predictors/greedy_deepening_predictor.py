from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from src.data.data_obj.feature_history import extend_history
import numpy as np
from src.usePydl.predictors.predictor import Predictor
import CONFIG


def generate_pred(sub_samples,n_total_samples,splits, weights) -> GreedyDeepeningPredictor:
    pred = GreedyDeepeningPredictor(sub_samples,n_total_samples,splits, weights)
    return pred


class GreedyDeepeningPredictor(Predictor):
    def __init__(self,sub_samples,n_total_samples,splits, weights):
        print(f"Starting Greedy Predictor, samples:{sub_samples.shape}")
        super().__init__(sub_samples,n_total_samples,splits, weights)
        print(f"Finished Predictor, samples:{sub_samples.shape}")


def build_tree_iteratively(feature_history, weights=None, n_worksers=10):
    samples = feature_history.samples
    n_samples = samples.shape[0]
    root_predictor = GreedyDeepeningPredictor(sub_samples=samples,n_total_samples=n_samples, splits=feature_history.get_splits_array(), weights=weights)
    feature_history.tree = root_predictor.tree
    future_to_leaf = {}

    def submit_leaves(history):
        print(f"Submitting leaves, depth:{history.depth}")
        for leaf in history.tree.get_leafs():
            sample_ids = leaf["value"].get("sample_ids", [])

            if len(sample_ids) >= CONFIG.MIN_SAMPLES_IN_LEAF:
                sub_samples = history.samples[sample_ids]
                sub_history = extend_history(sub_samples, history, leaf["value"]["leaf_id"])
                if weights is None:
                    if sub_history.depth < 2: new_weights = sub_history.get_feature_weights(mode='feat_imp')
                    else: new_weights = sub_history.get_feature_weights(mode='random')
                else: new_weights = weights
                sub_history.creat_splits(weights_each_feature=new_weights)

                future = executor.submit(generate_pred, sub_samples=sub_samples,n_total_samples=n_samples,splits=sub_history.get_splits_array(), weights=new_weights)
                future_to_leaf[future] = sub_history

    with ThreadPoolExecutor(max_workers=n_worksers) as executor:
        submit_leaves(feature_history)
        while future_to_leaf:
            done, _ = wait(future_to_leaf.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                current_history = future_to_leaf.pop(future)
                try:
                    new_pred = future.result()
                except Exception as exc:
                    print(f"Child predictor generated an exception: {exc}")
                    continue

                current_history.tree = new_pred.tree
                current_history.pred_error = new_pred.error
                if current_history.past is not None:
                    past_error = current_history.past.pred_error
                else: past_error = np.inf
                error_change = abs(past_error - new_pred.error)
                if current_history.depth < CONFIG.MAX_GREED_ITERATIONS and error_change > 1e-7:
                    print(f'error change! : {error_change}')
                    submit_leaves(current_history)

    return root_predictor
