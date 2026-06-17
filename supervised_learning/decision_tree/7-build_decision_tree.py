#!/usr/bin/env python3
""" Number of nodes/leaves in a decision tree."""
import numpy as np


class Node:
    """A node class containing leaves,roots."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Constructor of Node class."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth."""
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below() \
            if self.left_child else self.depth
        right = self.right_child.max_depth_below() \
            if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below, only leaves if specified."""
        if self.is_leaf:
            return 1

        if only_leaves:
            if self.left_child:
                left = self.left_child.count_nodes_below(True)
            else:
                left = 0
            if self.right_child:
                right = self.right_child.count_nodes_below(True)
            else:
                right = 0
            return left + right
        else:
            if self.left_child:
                left = self.left_child.count_nodes_below(False)
            else:
                left = 0
            if self.right_child:
                right = self.right_child.count_nodes_below(False)
            else:
                right = 0
            return 1 + left + right

    def get_leaves_below(self):
        """Return the list of all leaves of the tree."""
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update bounds for this node and recursively for all children."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def left_child_add_prefix(self, text):
        """Add ASCII prefix formatting for left child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add ASCII prefix formatting for right child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return an ASCII representation."""
        if self.is_root:
            label = (f"root [feature={self.feature}"
                     f", threshold={self.threshold}]")
        else:
            label = (f"node [feature={self.feature}"
                     f", threshold={self.threshold}]")

        result = label
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))\
                    .rstrip("\n")
        if self.right_child:
            result += "\n" +\
                    self.right_child_add_prefix(str(self.right_child))\
                    .rstrip("\n")
        return result

    def update_indicator(self):
        """Update the indicator function for this node."""

        def is_large_enough(x):
            """Check if all features are > lower bounds."""
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if key < x.shape[1]:
                    conditions.append(np.greater(x[:, key], self.lower[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
            if conditions:
                return np.all(np.array(conditions), axis=0)
            else:
                return np.ones(x.shape[0], dtype=bool)

        def is_small_enough(x):
            """Check if all features are <= upper bounds."""
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if key < x.shape[1]:
                    conditions.append(np.less_equal(x[:, key],
                                                    self.upper[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))

            if conditions:
                return np.all(np.array(conditions), axis=0)
            else:
                return np.ones(x.shape[0], dtype=bool)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                   is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Predict the value for a single sample x."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf class."""

    def __init__(self, value, depth=None):
        """Constructor of Leaf class."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the count of a leaf."""
        return 1

    def get_leaves_below(self):
        """Return the list containing this leaf."""
        return [self]

    def update_bounds_below(self):
        """Update bounds for leaf node - no action needed."""
        pass

    def __str__(self):
        """Return string representation of leaf node."""
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        """Predict the value for a single sample x."""
        return self.value


class Decision_Tree():
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Constructor of decision tree class."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the count of leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return the list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for the entire tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Update the prediction function for the decision tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Prediction function that uses leaf indicators."""
            n_individuals = A.shape[0]
            predictions = np.zeros(n_individuals, dtype=int)

            for leaf in leaves:
                mask = leaf.indicator(A)
                predictions[mask] = leaf.value

            return predictions

        self.predict = predict_func

    def pred(self, x):
        """Predict the value for a single sample x."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return minimum and maximum of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Generate a random split criterion (feature, threshold)."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree to training data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : """
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """Recursively fit a node by splitting or assigning leaves."""
        node.feature, node.threshold = self.split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        left_population = (node.sub_population &
                           (feature_values > node.threshold))
        right_population = (node.sub_population &
                            (feature_values <= node.threshold))

        left_count = np.sum(left_population)
        left_targets = self.target[left_population]
        is_left_leaf = (left_count < self.min_pop or
                        node.depth + 1 >= self.max_depth or
                        len(np.unique(left_targets)) == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        right_count = np.sum(right_population)
        right_targets = self.target[right_population]
        is_right_leaf = (right_count < self.min_pop or
                         node.depth + 1 >= self.max_depth or
                         len(np.unique(right_targets)) == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf node as a child of the given node."""
        targets = self.target[sub_population]
        values, counts = np.unique(targets, return_counts=True)
        value = values[np.argmax(counts)]

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a new internal node as a child of the given node."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy of decision tree on test data."""
        predictions = self.predict(test_explanatory)
        return np.sum(np.equal(predictions, test_target)) / test_target.size

    def __str__(self):
        """Return ASCII representation of the decision tree."""
        return self.root.__str__() + "\n"
