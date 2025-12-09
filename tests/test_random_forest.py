import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from Model.MyRandomForestClassifier import MyRandomForestClassifier
from Model.MyDecisionTree import MyDecisionTree as DecisionTree

# Basic fixtures
@pytest.fixture
def tiny_dataset():
    X = [
        [1, 2, 3],
        [1, 5, 2],
        [2, 1, 3],
        [2, 4, 1],
        [3, 1, 2],
        [3, 3, 1],
    ]
    y = ["A", "A", "B", "B", "C", "C"]
    return X, y


# Test 1: Initialization
def test_init():
    clf = MyRandomForestClassifier(N=10, M=3, F=2, random_seed=42)
    assert clf.N == 10
    assert clf.M == 3
    assert clf.F == 2


# Test 2: Stratified split keeps class balance
def test_stratified_split(tiny_dataset):
    X, y = tiny_dataset
    clf = MyRandomForestClassifier()

    X_train, y_train, X_test, y_test = clf._stratified_split(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    for cls in set(y):
        assert cls in y_train
        assert cls in y_test


# Test 3: Bootstrapping length preserved
def test_bootstrap(tiny_dataset):
    X, y = tiny_dataset
    clf = MyRandomForestClassifier()

    Xb, yb = clf._bootstrap(X, y)

    assert len(Xb) == len(X)
    assert len(yb) == len(y)

    assert len(set(tuple(x) for x in Xb)) <= len(X)


# Test 4: Feature selection
def test_feature_subset():
    clf = MyRandomForestClassifier(F=2)

    features = clf._select_features(total_features=5)
    
    assert len(features) == 2
    assert all(0 <= f < 5 for f in features)
    assert features == sorted(features)


# Test 5: fit() trains a forest
def test_fit_creates_forest(tiny_dataset):
    X, y = tiny_dataset
    clf = MyRandomForestClassifier(N=5, M=2, F=2)

    clf.fit(X, y)

    assert len(clf.forest) == 2  
    for tree, feat in clf.forest:
        assert isinstance(tree, DecisionTree)
        assert isinstance(feat, list)

# Test 6: predict() outputs labels of correct type
def test_predict(tiny_dataset):
    X, y = tiny_dataset
    clf = MyRandomForestClassifier(N=6, M=3, F=2)

    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == len(X)
    for p in preds:
        assert p in y
