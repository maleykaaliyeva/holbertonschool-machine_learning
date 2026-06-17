#!/usr/bin/env python3
"""
Module to build a decision tree from scratch with custom string representation.
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively calculates the maximum depth of the tree below this node.
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts nodes below this node.
        """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """
        Adds prefixes to the left child string layout.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |      " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds prefixes to the right child string layout.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("           " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Returns string layout block of the node and sub-nodes.
        """
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        out += self.left_child_add_prefix(str(self.left_child))
        out += self.right_child_add_prefix(str(self.right_child))
        return out.rstrip("\n")


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        """Initializes a Leaf instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since a leaf is a single node.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.
        """
        return f"leaf [value={self.value}]"


class Decision_Tree():
    """
    Represents a Decision Tree classifier/regressor.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision Tree instance."""
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
        """
        Returns the maximum depth of the decision tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the number of nodes in the decision tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns the complete structural string representation of the tree.
        """
        return self.root.__str__() + "\n"
