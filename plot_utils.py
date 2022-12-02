"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains reusable plotting functions
    that can be used in the Jupyter Notebooks.
"""
import matplotlib.pyplot as plt
def plot_histogram(data, x_label, y_label, title):
    """Displayes a histogram of the given data.

    Args:
        data(list): list of data
        x_label(str): label for the x axis
        y_label(str): label for the y axis
        title(str): title of the chart
    """
    plt.figure()
    plt.hist(data, color=(0, 0.5, 1.0)) # default bins is 10
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def box_plot(distributions, labels, title):
    """Displayes a box plot of the given x and y data.

    Args:
        distributions(list): list of values
        labels(list): list of labels for the distribution data
        title(str): title of the chart
    """
    plt.figure()
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(distributions) + 1)), labels)
    plt.title(title)
    plt.show()

def bar_chart_example(x, y):
    plt.figure()
    plt.bar(x, y)
    plt.show()


