#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
"""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_data():
    """
    Loads and splits dataset into train, validation, and test sets
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_state=42, test_size=0.3
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def optimize_hyperparameters():
    """
    Optimizes a Multi-Layer Perceptron classifier using GPyOpt across
    5 hyperparameters: learning rate, hidden layer units, L2 penalty (alpha),
    batch size, and max iterations (epochs).
    """
    X_train, y_train, X_val, y_val, _, _ = load_data()

    # Define 5 Hyperparameters Search Domain
    domain = [
        {'name': 'learning_rate_init', 'type': 'continuous',
         'domain': (1e-4, 1e-1)},
        {'name': 'hidden_layer_sizes', 'type': 'discrete',
         'domain': (16, 32, 64, 128)},
        {'name': 'alpha', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)},
        {'name': 'max_iter', 'type': 'discrete', 'domain': (50, 100, 200)}
    ]

    best_score = float('inf')

    def objective_function(x):
        """
        Objective function to minimize (1 - validation_accuracy)
        """
        nonlocal best_score

        # Extract hyperparameters from optimizer input
        params = x[0]
        lr = float(params[0])
        units = int(params[1])
        alpha = float(params[2])
        b_size = int(params[3])
        epochs = int(params[4])

        # Initialize MLPClassifier with early stopping
        mlp = MLPClassifier(
            hidden_layer_sizes=(units,),
            learning_rate_init=lr,
            alpha=alpha,
            batch_size=b_size,
            max_iter=epochs,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42
        )

        # Train model
        mlp.fit(X_train, y_train)

        # Satisficing metric: Validation Accuracy
        val_acc = mlp.score(X_val, y_val)
        loss = 1.0 - val_acc

        # Save checkpoint of best iteration
        if loss < best_score:
            best_score = loss
            filename = (
                f"checkpoint_lr{lr:.4f}_units{units}_alpha{alpha:.5f}_"
                f"bs{b_size}_epochs{epochs}.npz"
            )
            np.savez(
                filename,
                coefs=mlp.coefs_,
                intercepts=mlp.intercepts_
            )

        return loss

    # Initialize Bayesian Optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=domain,
        acquisition_type='EI',
        exact_feval=True
    )

    # Run optimization for maximum 30 iterations
    optimizer.run_optimization(max_iter=30)

    # Plot convergence graph and save it
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')
    plt.close()

    # Save optimization report to file
    best_x = optimizer.x_opt
    best_y = 1.0 - optimizer.fx_opt

    report = (
        "Bayesian Optimization Report\n"
        "============================\n"
        f"Best Validation Accuracy (Satisficing Metric): {best_y:.4f}\n\n"
        "Optimal Hyperparameters:\n"
        f"  - Learning Rate: {best_x[0]:.6f}\n"
        f"  - Hidden Layer Units: {int(best_x[1])}\n"
        f"  - L2 Regularization (Alpha): {best_x[2]:.6f}\n"
        f"  - Batch Size: {int(best_x[3])}\n"
        f"  - Epochs (Max Iterations): {int(best_x[4])}\n"
    )

    with open('bayes_opt.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    optimize_hyperparameters()
