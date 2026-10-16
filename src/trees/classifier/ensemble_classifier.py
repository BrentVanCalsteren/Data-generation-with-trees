import numpy as np
from pydl85 import DL85Classifier

class EnsembleClassifier:
    #will split into n different ensembles based on the number of different unique labels
    def __init__(self, max_depth=3, min_sup=1, time_limit=100):
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.time_limit = time_limit
        self.classifiers = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            print(f"Training for recognizing value: {c}")
            y_bin = (y == c).astype(int)
            clf = DL85Classifier(max_depth=self.max_depth, min_sup=self.min_sup, time_limit=self.time_limit)
            clf.fit(X, y_bin)
            self.classifiers[c] = clf

    def predict(self, X):
        # We will collect the probability for each class
        predictions = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            clf = self.classifiers[c]
            if hasattr(clf, 'predict_proba'):
                try:
                    probs = clf.predict_proba(X)
                    if probs.shape[1] > 1:
                        predictions[:, idx] = probs[:, 1]
                    else:
                        predictions[:, idx] = clf.predict(X)
                except Exception:
                    predictions[:, idx] = clf.predict(X)
            else:
                predictions[:, idx] = clf.predict(X)

        best_indices = np.argmax(predictions, axis=1)
        return self.classes[best_indices]
