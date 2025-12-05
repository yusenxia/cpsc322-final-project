import random
from collections import Counter
from Model.Decision_Tree import DecisionTree   

class RandomForest:

    def __init__(self, n_trees=15, max_features=None, random_seed=0):
        self.n_trees = n_trees
        self.max_features = max_features
        self.forest = []
        random.seed(random_seed)

    def _bootstrap_sample(self, X, y):
        n = len(X)
        indices = [random.randint(0, n - 1) for _ in range(n)]
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def _select_features(self, n_features):
        if self.max_features is None:
            k = max(1, int(n_features ** 0.5))  
        else:
            k = self.max_features

        return sorted(random.sample(range(n_features), k))

    def fit(self, X, y):
        self.forest = []
        n_features = len(X[0])

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            feature_subset = self._select_features(n_features)

            X_reduced = [[row[i] for i in feature_subset] for row in X_sample]

            tree = DecisionTree()
            tree.fit(X_reduced, y_sample)

            self.forest.append((tree, feature_subset))

    def _predict_one(self, x):
        votes = []
        for tree, feature_subset in self.forest:
            x_reduced = [x[i] for i in feature_subset]
            pred = tree.predict([x_reduced])[0]
            votes.append(pred)

        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict_one(x) for x in X]
