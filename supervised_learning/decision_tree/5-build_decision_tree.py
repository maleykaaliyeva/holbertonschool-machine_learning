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
        """Add prefix for left child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child.."""
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

        self.indicator = lambda x: np.all(np.array(
                                                   [is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)


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
        return f"-> leaf [value={self.value}]"


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

    def __str__(self):
        """Print the tree."""
        return self.root.__str__() + "\n"
