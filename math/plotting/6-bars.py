#!/usr/bin/env python3
"""This module defines bars function."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Show stacked bars for different people."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ["Farrah", "Fred", "Felicia"]
    fruits = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    x = np.arange(len(people))
    bottom = np.zeros(len(people))

    for i in range(len(fruits)):
        plt.bar(x, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=fruits[i])
        bottom += fruit[i]

    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
