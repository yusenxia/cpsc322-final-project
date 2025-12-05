import math
import random
from collections import Counter

#  Helper functions

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())


def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]


def make_bins(values, k=3):
    mn, mx = min(values), max(values)
    width = (mx - mn) / k
    return [(mn + i*width, mn + (i+1)*width) for i in range(k)]


def assign_bin(value, bins):
    for i, (low, high) in enumerate(bins):
        if low <= value <= high:
            return f"bin{i}"
    return f"bin{len(bins)-1}"


class DecisionTree:
    def __init__(self):
        self.tree = None

    # Training

    def fit(self, X, y):
        data = [row + [label] for row, label in zip(X, y)]
        attributes = list(range(len(X[0])))
        self.tree = self._tdidt(data, attributes)

    def _tdidt(self, data, attributes):
        labels = [row[-1] for row in data]

        if len(set(labels)) == 1:
            return ("Leaf", labels[0])

        if not attributes:
            return ("Leaf", majority_vote(labels))

        best_attr = None
        best_gain = -999

        for attr in attributes:
            gain = self._information_gain(data, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        partitions = {}
        for row in data:
            val = row[best_attr]
            partitions.setdefault(val, []).append(row)

        new_attrs = [a for a in attributes if a != best_attr]

        tree = ("Node", best_attr, {})

        for val, subset in partitions.items():
            subtree = self._tdidt(subset, new_attrs)
            tree[2][val] = subtree

        return tree

    def _information_gain(self, data, attr):
        parent_labels = [row[-1] for row in data]
        parent_entropy = entropy(parent_labels)

        partitions = {}
        for row in data:
            partitions.setdefault(row[attr], []).append(row)

        weighted_entropy = 0
        total = len(data)

        for subset in partitions.values():
            subset_labels = [row[-1] for row in subset]
            weighted_entropy += (len(subset) / total) * entropy(subset_labels)

        return parent_entropy - weighted_entropy

    # Prediction
    def predict_one(self, row):
        node = self.tree
        while node[0] != "Leaf":
            _, attr, branches = node
            value = row[attr]

            if value in branches:
                node = branches[value]
            else:
                return random.choice(list(branches.values()))[1]

        return node[1]

    def predict(self, X):
        return [self.predict_one(row) for row in X]


#  Evaluation

def accuracy_score(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision_recall_f1(y_true, y_pred, positive_label):
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


#  10-fold Cross Validation

def k_fold_split(X, y, k=10, seed=0):
    random.seed(seed)

    n = len(X)
    idxs = list(range(n))
    random.shuffle(idxs)

    fold_size = n // k
    folds = [idxs[i*fold_size : (i+1)*fold_size] for i in range(k-1)]
    folds.append(idxs[(k-1)*fold_size :])

    fold_sets = []
    for i in range(k):
        test = folds[i]
        train = [idx for j in range(k) if j != i for idx in folds[j]]

        X_train = [X[j] for j in train]
        y_train = [y[j] for j in train]
        X_test  = [X[j] for j in test]
        y_test  = [y[j] for j in test]

        fold_sets.append((X_train, y_train, X_test, y_test))

    return fold_sets
