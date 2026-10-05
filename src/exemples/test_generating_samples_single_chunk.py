from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from data.data_obj.sampels import Samples
import numpy as np
from pydl85 import DL85Classifier
from src.usePydl.classifier.ensemble_classifier import EnsembleClassifier
from sklearn.metrics import accuracy_score, classification_report
from usePydl.predictors.ensemble_predictors import (RandomForestPredictor,
                                                    LeafCoherenceBoostingPredictor, GANBoostingPredictor)
from usePydl.predictors.data_gen_methods import (gen_new_data_guided,
                        gen_new_data_by_chain_feat_sampling, gen_data_from_single_tree, gen_new_data_gibbs)
from usePydl.predictors.feat_conditional_predictor import FeatConditionalPredictor
from src.data.data_obj.feature_history import FeatureHistory, get_feature_weights
from src.evalData.eval_data import evaluate

DO_CLASSIFICATION = False
def test_data_generation():
    sample_obj = Samples(dataset='iris' ,data_type='tabular',labels_at_front=False)
    #sample_obj = Samples(dataset='mnist_jpeg', data_type='image', labels_at_front=False)
    sample_obj.load_chunk(chunk_id=0,split_test=0.2)
    samples_train = sample_obj.samples
    samples_test = sample_obj.samples_test
    sample_obj.save_output(samples=samples_train,llables=None,output_name='original_encoded')

    #==========================
    #create splits
    feature_data = FeatureHistory(samples=samples_train)
    w = None
    feature_data.creat_splits(w)
    train_splits = feature_data.get_splits_array().T
    test_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_test).T
    same_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_train).T
    print(f'bool convertion works correctly: {np.equal(train_splits, same_splits).all()}')


    #=========================================
    #now let's test the quality of the splits, good splits will result into good classification with dlclassifier
    labels_train = sample_obj.labels.flatten()
    labels_test = sample_obj.labels_test.flatten()
    classify(train_splits,test_splits,labels_train,labels_test)
    #=================================================
    #now let's generate new data
    #pred = build_tree_iteratively(feature_data)
    #samples_gen = pred.gen_data_from_single_tree(n=200, conf=1)
    #pred = RandomForestPredictor(sample_obj)
    #pred = LeafCoherenceBoostingPredictor(sample_obj=sample_obj, n_trees=10*samples_train.shape[1])
    pred = FeatConditionalPredictor(sample_obj=sample_obj)
    #samples_gen = gen_new_data_by_chain_feat_sampling(pred,n=200)
    samples_gen = gen_new_data_guided(predictor=pred,n=400)
    #samples_gen = gen_new_data_by_chain_feat_sampling(predictor=pred,n=400)
    #samples_gen = gen_new_data_gibbs(predictor=pred,n=400)

    #=================================================
    #now let's see if classification is better with extra generated data
    feature_data = FeatureHistory(samples=samples_gen)
    feature_data.creat_splits(w)
    train_splits = feature_data.get_splits_array().T
    test_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_test).T

    labels_gen = sample_obj.get_best_matching_labels(gen_samples=samples_gen, chunk_id=0)
    #test on generated data alone
    classify(train_splits, test_splits, labels_gen, labels_test)

    samples_combined = np.vstack((samples_train, samples_gen))
    feature_data = FeatureHistory(samples=samples_combined)
    feature_data.creat_splits(w)
    train_splits = feature_data.get_splits_array().T
    test_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_test).T
    y_combined = np.hstack((labels_train, labels_gen))
    # test on combined data
    classify(train_splits, test_splits, y_combined, labels_test)
    #All eval metrics
    evaluate(real_samples=samples_train, real_labels=labels_train, gen_samples=samples_gen,gen_labels=labels_gen,test_samples=samples_test,sample_obj=sample_obj)
    sample_obj.save_output(samples=samples_gen, llables=labels_gen,output_name='generated')


def classify_ensemble(x_train, x_test, y_train, y_test):
    if DO_CLASSIFICATION:
        print("Running pydl classifier")
        clasfi = EnsembleClassifier(max_depth=3, min_sup=1, time_limit=100)
        clasfi.fit(x_train,y_train)
        y_pred_test = clasfi.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

def classify(x_train, x_test, y_train, y_test):
    if DO_CLASSIFICATION:
        depth = 1
        uniques = np.unique(y_train)
        while 2**(depth-1) < len(uniques):
            depth += 1
        print("Running pydl classifier")
        clasfi = create_classifier_default(
            x_bin=x_train, y=y_train, max_depth=depth, min_sup=1, time=100)
        y_pred_test = clasfi.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

def create_classifier_default(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi

def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array==val] = i
    return indeces



if __name__ == "__main__":
    test_data_generation()
