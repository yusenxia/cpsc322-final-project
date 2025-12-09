import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from Model.MyDecisionTree import MyDecisionTree


X_small = [
    [1, 0],
    [1, 1],
    [2, 0],
    [2, 1],
    [9, 0],
    [9, 1]
]

y_small = ["A", "A", "A", "A", "B", "B"]


def test_fit_runs():
    tree = MyDecisionTree()
    tree.fit(X_small, y_small)

    assert tree.tree is not None, "Tree should be built after fit()."
    assert isinstance(tree.tree, list), "Tree structure must be a list."


def test_predict_length():
    tree = MyDecisionTree()
    tree.fit(X_small, y_small)
    preds = tree.predict(X_small)

    assert len(preds) == len(X_small), "Predict should match input size."


def test_predict_labels_valid():
    tree = MyDecisionTree()
    tree.fit(X_small, y_small)
    preds = tree.predict(X_small)

    valid = set(y_small)
    assert all(p in valid for p in preds), "Predictions must exist in training labels."


def test_simple_separable_dataset():
    X = [[1], [1], [2], [2], [9], [9]]
    y = ["A", "A", "A", "A", "B", "B"]

    tree = MyDecisionTree()
    tree.fit(X, y)
    preds = tree.predict(X)

    assert preds[:4].count("A") >= 4, "First 4 should be predicted as A."
    assert preds[4:].count("B") >= 2, "Last 2 should be predicted as B."


def test_default_label_used():
    X_train = [[1], [2], [3]]
    y_train = ["A", "A", "B"]

    tree = MyDecisionTree()
    tree.fit(X_train, y_train)

    pred = tree.predict([[999]])[0]

    assert pred == tree.default_label, "Should return default label for unseen attribute value."


def test_random_feature_subset():
    tree = MyDecisionTree(F=1)  
    tree.fit(X_small, y_small)

    root = tree.tree
    assert root[0] == "Attribute"
    root_attr = root[1]

    assert root_attr in ("att0", "att1"), "Random F selection should restrict candidates."


