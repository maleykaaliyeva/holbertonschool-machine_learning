#!/usr/bin/env python3
"""This module defines frequency function."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Show histogram of grades."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.axis([0, 100, 0, 30])
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor="k")
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
