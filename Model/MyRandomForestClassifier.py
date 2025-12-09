import random
from collections import Counter
from Model.MyDecisionTree import DecisionTree


class MyRandomForestClassifier:

    def __init__(self, N=10, M=5, F=2, random_seed=0):
        self.N = N                
        self.M = M                
        self.F = F               
        self.forest = []          
        random.seed(random_seed)

    # Helper: stratified split (1/3 test, 2/3 remainder)
    def _stratified_split(self, X, y):
        class_groups = {}
        for xi, yi in zip(X, y):
            class_groups.setdefault(yi, []).append(xi)

        X_train, y_train, X_test, y_test = [], [], [], []

        for cls, xs in class_groups.items():
            n = len(xs)
            test_size = max(1, n // 3)

            random.shuffle(xs)

            test_x = xs[:test_size]
            train_x = xs[test_size:]

            X_test.extend(test_x)
            y_test.extend([cls] * len(test_x))

            X_train.extend(train_x)
            y_train.extend([cls] * len(train_x))

        return X_train, y_train, X_test, y_test

    # Bootstrapping
    def _bootstrap(self, X, y):
        n = len(X)
        idx = [random.randint(0, n - 1) for _ in range(n)]
        return [X[i] for i in idx], [y[i] for i in idx]

    # Random feature subset (per tree)
    def _select_features(self, total_features):
        k = min(self.F, total_features)
        return sorted(random.sample(range(total_features), k))

    def fit(self, X, y):
        X_remain, y_remain, X_test, y_test = self._stratified_split(X, y)

        total_features = len(X[0])
        candidate_trees = []

        for _ in range(self.N):

            X_sample, y_sample = self._bootstrap(X_remain, y_remain)

            feature_subset = self._select_features(total_features)

            X_reduced = [[row[i] for i in feature_subset] for row in X_sample]

            tree = DecisionTree(F=self.F)   
            tree.fit(X_reduced, y_sample)

            X_val = [[row[i] for i in feature_subset] for row in X_test]
            preds = tree.predict(X_val)

            correct = sum(p == t for p, t in zip(preds, y_test))
            acc = correct / len(y_test)

            candidate_trees.append((acc, tree, feature_subset))

        candidate_trees.sort(reverse=True, key=lambda x: x[0])
        self.forest = [(tree, feat) for _, tree, feat in candidate_trees[:self.M]]

    # Predict one sample
    def _predict_one(self, x):
        votes = []
        for tree, feat in self.forest:
            x_reduced = [x[i] for i in feat]
            pred = tree.predict([x_reduced])[0]
            votes.append(pred)

        return Counter(votes).most_common(1)[0][0]

    # Predict
    def predict(self, X):
        return [self._predict_one(row) for row in X]

