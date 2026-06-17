#!/usr/bin/env python3
"""Building a Decision Tree."""
import numpy as np


class Node:
    """A node class that generalizes everything including root and leaves."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Construct the Node object."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Find the maximum depth."""
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below()\
            if self.left_child else self.depth
        right = self.right_child.max_depth_below()\
            if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below, only leaves if specified."""
        if self.is_leaf:
            return 1

        if only_leaves:
            left = self.left_child.count_nodes_below(True)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(True)\
                if self.right_child else 0
            return left + right
        else:
            left = self.left_child.count_nodes_below(False)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(False)\
                if self.right_child else 0
            return 1 + left + right

    def __str__(self):
        """Return an ASCII representation of the tree from this node."""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            s = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            left_str = self.left_child.__str__()
            s += "\n" + self.left_child_add_prefix(left_str).rstrip("\n")

        if self.right_child:
            right_str = self.right_child.__str__()
            s += "\n" + self.right_child_add_prefix(right_str).rstrip("\n")

        return s

    def left_child_add_prefix(self, text):
        """Add ASCII branch prefixes for a left child in a tree diagram."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add ASCII branch prefixes for a right child in a tree diagram."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Return the leaves of a node."""
        if self.is_leaf:
            return [self]

        list_of_leaves = []
        if self.left_child:
            list_of_leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            list_of_leaves.extend(self.right_child.get_leaves_below())
        return list_of_leaves

    def update_bounds_below(self):
        """Recursively compute bounds for each node (single feature)."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            feature = self.feature
            threshold = self.threshold

            if child is self.left_child:
                child.lower[feature] = threshold
            else:
                child.upper[feature] = threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Return an indicator array into indicator attribute."""

        def is_large_enough(x):
            """Check ith individual has all ftrs bigger than lower bounds."""
            checks = [x[:, key] > self.lower[key]
                      for key in self.lower.keys()]
            return np.all(np.array(checks), axis=0)

        def is_small_enough(x):
            """Check ith individual has all ftrs less than upper bounds."""
            checks = [x[:, key] <= self.upper[key]
                      for key in self.upper.keys()]
            return np.all(np.array(checks), axis=0)

        self.indicator = lambda x: \
            np.logical_and(is_large_enough(x), is_small_enough(x))

    def pred(self, x):
        """Predict for a single node."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Terminal node which is a leaf."""

    def __init__(self, value, depth=None):
        """Construct the leaf object."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the count of 1 leaf."""
        return 1

    def __str__(self):
        """Print the ASCII representation of a leaf."""
        return f"leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return the leaf object."""
        return [self]

    def update_bounds_below(self):
        """Bound of a leaf."""
        pass

    def pred(self, x):
        """Return the value of leaf as a prediction."""
        return self.value


class Decision_Tree():
    """The whole Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Construct the decision tree."""
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

    def __str__(self):
        """Print the whole tree in ASCII."""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Return the leaves of the decision tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Bounds of the whole tree."""
        self.root.update_bounds_below()

    def pred(self, x):
        """Make a prediction for whole tree."""
        return self.root.pred(x)

    def update_predict(self):
        """Vectorize the predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        # Vectorized prediction using numpy operations
        self.predict = lambda A: np.array([leaf.value for leaf in leaves])[
            np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)
        ]

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree model."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        a1, a2 = self.explanatory, self.target
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(a1, a2)    }""")

    def np_extrema(self, arr):
        """Return the minimum and maximum of the array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Split the population based on random criterion."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            column_values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(column_values)
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively split node into children until leaves are reached."""
        # Get target values for individuals in this node's sub_population
        y_node = self.target[node.sub_population]

        # Check if node should be a leaf:
        # 1. Less than min_pop individuals
        # 2. Reached max_depth
        # 3. All individuals have same target value (pure class)
        if (node.sub_population.sum() < self.min_pop or
            node.depth >= self.max_depth or
                len(np.unique(y_node)) == 1):

            # Make this node a leaf with most represented class
            values, counts = np.unique(y_node, return_counts=True)
            most_frequent_class = values[np.argmax(counts)]

            # Convert node to leaf by setting leaf properties
            node.is_leaf = True
            node.value = most_frequent_class
            node.feature = None
            node.threshold = None
            return

        # Get split criterion (feature and threshold)
        node.feature, node.threshold = self.split_criterion(node)

        # Handle case where no valid split found (infinite threshold)
        if np.isinf(node.threshold):
            values, counts = np.unique(y_node, return_counts=True)
            most_frequent_class = values[np.argmax(counts)]
            node.is_leaf = True
            node.value = most_frequent_class
            node.feature = None
            node.threshold = None
            return

        # Vectorized split
        feature_values = self.explanatory[node.sub_population, node.feature]
        left_mask_local = feature_values > node.threshold
        right_mask_local = ~left_mask_local

        # Create global masks for left and right children
        left_population = np.zeros_like(self.target, dtype=bool)
        right_population = np.zeros_like(self.target, dtype=bool)

        # Map local masks to global indices
        left_population[node.sub_population] = left_mask_local
        right_population[node.sub_population] = right_mask_local

        # Safety check: ensure split actually separates data
        if left_population.sum() == 0 or right_population.sum() == 0:
            values, counts = np.unique(y_node, return_counts=True)
            most_frequent_class = values[np.argmax(counts)]
            node.is_leaf = True
            node.value = most_frequent_class
            node.feature = None
            node.threshold = None
            return

        # Create and fit left child
        node.left_child = Node(depth=node.depth + 1)
        node.left_child.sub_population = left_population
        self.fit_node(node.left_child)

        # Create and fit right child
        node.right_child = Node(depth=node.depth + 1)
        node.right_child.sub_population = right_population
        self.fit_node(node.right_child)

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy score on test data."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size

    def possible_thresholds(self, node, feature):
        """Return possible split thresholds for a given node and feature."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1])/2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Compute best threshold and Gini score for a single feature."""
        # Get feature values and target values for individuals in this node
        x_node = self.explanatory[node.sub_population, feature]
        y_node = self.target[node.sub_population]

        # Get possible thresholds
        thresholds = self.possible_thresholds(node, feature)

        # If no valid thresholds (all values same), return high Gini
        if len(thresholds) == 0:
            return 0, np.inf

        # Get unique classes in this node
        classes = np.unique(y_node)

        # Create Left_F array of shape (n, t, c) where:
        # n = number of individuals in sub_population
        # t = number of possible thresholds
        # c = number of classes
        # Left_F[i, j, k] = True iff i-th individual is of class k AND
        #                   feature value > j-th threshold

        # First create boolean arrays for each condition
        # Shape: (n, t) - True if individual i has feature value > threshold j
        feature_condition = x_node[:, np.newaxis] > thresholds[np.newaxis, :]

        # Shape: (n, c) - True if individual i belongs to class k
        class_condition = y_node[:, np.newaxis] == classes[np.newaxis, :]

        # Combine conditions using broadcasting to get shape (n, t, c)
        # Left_F[i, j, k] = feature_condition[i, j] AND class_condition[i, k]
        Left_F = feature_condition[:, :, np.newaxis] &\
            class_condition[:, np.newaxis, :]

        feature_condition_right = x_node[:, np.newaxis]\
            <= thresholds[np.newaxis, :]
        Right_F = feature_condition_right[:, :, np.newaxis] &\
            class_condition[:, np.newaxis, :]

        # Sum over individuals (axis=0) to get class counts for each threshold
        left_counts = Left_F.sum(axis=0)   # shape (t, c)
        right_counts = Right_F.sum(axis=0)  # shape (t, c)

        # Total individuals in left and right for each threshold
        left_totals = left_counts.sum(axis=1)   # shape (t,)
        right_totals = right_counts.sum(axis=1)  # shape (t,)

        # Compute Gini impurity for left and right children
        # Gini = 1 - sum(p_k^2) where p_k = class_count / total_count

        # Handle empty children (avoid division by zero)
        left_totals_safe = np.where(left_totals == 0, 1, left_totals)
        right_totals_safe = np.where(right_totals == 0, 1, right_totals)

        # Compute proportions
        left_proportions = left_counts / left_totals_safe[:, np.newaxis]
        right_proportions = right_counts / right_totals_safe[:, np.newaxis]

        # Compute Gini impurity: 1 - sum(p_k^2)
        left_gini = 1 - np.sum(left_proportions ** 2, axis=1)  # shape (t,)
        right_gini = 1 - np.sum(right_proportions ** 2, axis=1)  # shape (t,)

        # Set Gini to 0 for empty children (perfectly pure by definition)
        left_gini = np.where(left_totals == 0, 0, left_gini)
        right_gini = np.where(right_totals == 0, 0, right_gini)

        # Compute weighted average Gini impurity for each threshold
        total_size = left_totals + right_totals

        # Weighted average
        sum_gini = left_gini * left_totals + right_gini * right_totals
        avg_gini = sum_gini / total_size

        # Find threshold with minimum average Gini
        best_idx = np.argmin(avg_gini)

        return thresholds[best_idx], avg_gini[best_idx]

    def Gini_split_criterion(self, node):
        """Find feature and threshold with lowest Gini impurity for a node."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                     for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
