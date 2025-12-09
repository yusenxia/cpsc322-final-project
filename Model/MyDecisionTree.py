import math
import random
from collections import Counter

class MyDecisionTree:

    def __init__(self, F=None):
        self.tree = None
        self.F = F      
        self.default_label = None  

    # ---------- Helper Functions ----------

    def _entropy(self, labels):
        total = len(labels)
        freq = Counter(labels)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _majority_label(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def _combine_instances(self, X, y):
        return [row[:] + [label] for row, label in zip(X, y)]

    def _tdidt(self, instances, available_attributes, parent_partition_size):
        labels = [row[-1] for row in instances]

        if len(set(labels)) == 1:
            return ["Leaf", labels[0], len(instances), parent_partition_size]

        if not available_attributes:
            maj = self._majority_label(labels)
            return ["Leaf", maj, len(instances), parent_partition_size]

        if self.F is not None:
            attributes_to_consider = random.sample(
                available_attributes,
                min(self.F, len(available_attributes))
            )
        else:
            attributes_to_consider = available_attributes[:]

        # select best attribute using information gain
        parent_entropy = self._entropy(labels)
        best_attr = None
        best_gain = -1

        for attr in attributes_to_consider:
            partitions = {}
            for row in instances:
                partitions.setdefault(row[attr], []).append(row)

            weighted_entropy = 0
            total = len(instances)
            for subset in partitions.values():
                subset_labels = [row[-1] for row in subset]
                weighted_entropy += (len(subset)/total) * self._entropy(subset_labels)

            gain = parent_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        node = ["Attribute", f"att{best_attr}"]

        new_available = [a for a in available_attributes if a != best_attr]

        partitions = {}
        for row in instances:
            partitions.setdefault(row[best_attr], []).append(row)

        for val, subset in partitions.items():
            child = self._tdidt(subset, new_available, len(instances))
            node.append(["Value", val, child])

        return node

    # ---------- Public fit() ----------

    def fit(self, X_train, y_train):
        instances = self._combine_instances(X_train, y_train)
        n_features = len(X_train[0])
        attributes = list(range(n_features))
        self.default_label = self._majority_label(y_train)
        self.tree = self._tdidt(instances, attributes, len(instances))

    # ---------- Prediction ----------

    def _predict_one(self, x):
        node = self.tree

        while node[0] != "Leaf":
            attr_tag = node[1]
            attr_index = int(attr_tag[3:])
            x_val = x[attr_index]

            child_found = False
            for branch in node[2:]:
                if branch[1] == x_val:
                    node = branch[2]
                    child_found = True
                    break

            if not child_found:
                return self.default_label

        return node[1]

    def predict(self, X_test):
        preds = []
        for x in X_test:
            pred = self._predict_one(x)
            preds.append(pred)
        return preds

DecisionTree = MyDecisionTree