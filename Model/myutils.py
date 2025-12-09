import random
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from Model.MyDecisionTree import MyDecisionTree
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Binning helpers 
def make_bins(col, k=5):

    col = np.asarray(col, dtype=float)
    mn = col.min()
    mx = col.max()

    if mn == mx:
        return np.zeros_like(col, dtype=int)

    width = (mx - mn) / k
    bins = np.floor((col - mn) / width).astype(int)

    bins = np.clip(bins, 0, k - 1)
    return bins


def bin_numeric_df(df, numeric_cols, k=5, as_str=True):
    df_binned = df.copy()
    for c in numeric_cols:
        b = make_bins(df_binned[c].to_numpy(), k=k)
        if as_str:
            df_binned[c] = b.astype(str)
        else:
            df_binned[c] = b
    return df_binned

# Cross-validation helper 
def run_tree_cv(X, y, classifier_class, k=10, random_state=0):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    metrics_list = []

    for train_idx, test_idx in kf.split(X):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        model = classifier_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        err  = 1 - acc
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        metrics_list.append((acc, err, prec, rec, f1, cm))

    return metrics_list

#  Random STRATIFIED split 
def stratified_split(X, y, test_ratio=0.33, seed=0):
    random.seed(seed)
    X = np.array(X)
    y = np.array(y)

    labels = np.unique(y)
    test_idx = []
    remainder_idx = []

    for lbl in labels:
        idx = np.where(y == lbl)[0].tolist()
        random.shuffle(idx)

        cut = int(len(idx) * test_ratio)
        test_idx.extend(idx[:cut])
        remainder_idx.extend(idx[cut:])

    return (
        X[test_idx].tolist(),
        y[test_idx].tolist(),
        X[remainder_idx].tolist(),
        y[remainder_idx].tolist(),
    )

# Bootstrap sample from remainder set
def bootstrap_sample(X, y):
    n = len(X)
    idx = [random.randint(0, n - 1) for _ in range(n)]
    Xs = [X[i] for i in idx]
    ys = [y[i] for i in idx]
    return Xs, ys


# Randomly select F features 
def select_random_features(n_features, F):
    F = min(F, n_features)
    return sorted(random.sample(range(n_features), F))


# Train 1 random tree using feature subset
def train_random_decision_tree(X, y, feature_idx):
    X_reduced = [[row[i] for i in feature_idx] for row in X]

    tree = MyDecisionTree()
    tree.fit(X_reduced, y)
    return tree

# Evaluate a tree on validation set
def evaluate_tree(tree, X_val, y_val, feature_idx):
    X_val_reduced = [[row[i] for i in feature_idx] for row in X_val]
    y_pred = tree.predict(X_val_reduced)
    return accuracy_score(y_val, y_pred)


# Full Random Forest (teacher-defined version)
def run_random_forest(
    X_train, y_train, X_test, y_test,
    N=20, M=7, F=2, seed=0
):
    random.seed(seed)
    n_features = len(X_train[0])

    trained_trees = []

    for _ in range(N):
        X_bs, y_bs = bootstrap_sample(X_train, y_train)

        feature_idx = select_random_features(n_features, F)

        tree = train_random_decision_tree(X_bs, y_bs, feature_idx)

        val_acc = evaluate_tree(tree, X_train, y_train, feature_idx)

        trained_trees.append((val_acc, tree, feature_idx))

    trained_trees.sort(reverse=True, key=lambda x: x[0])
    selected = trained_trees[:M]

    predictions = []
    for x in X_test:
        votes = []
        for _, tree, feats in selected:
            x_reduced = [x[i] for i in feats]
            pred = tree.predict([x_reduced])[0]
            votes.append(pred)

        final = Counter(votes).most_common(1)[0][0]
        predictions.append(final)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions, selected