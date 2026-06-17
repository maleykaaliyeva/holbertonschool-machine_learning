#!/usr/bin/env python3
"""Random Forest Algorithm."""
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree  # noqa: E402
import numpy as np


class Random_Forest():
    """Random FOrest class."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize random forest object."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict the class for each example by majority vote of trees."""
        # Collect predictions from each tree (n_trees × n_samples)
        prdctions = np.array([tree(explanatory) for tree in self.numpy_preds])

        # Take the majority vote along axis=0 (over trees, per sample)
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                     axis=0, arr=prdctions)

        return y_pred

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fit the model."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        a1, a2 = self.explanatory, self.target
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : {self.accuracy(a1, a2)}""")

    def accuracy(self, test_explanatory, test_target):
        """Measure the accuracy of the algorithm."""
        acc = np.sum(np.equal(self.predict(test_explanatory), test_target))
        acc = acc / test_target.size
        return acc
