"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains reusable plotting functions
    that can be used in the Jupyter Notebooks.
"""
import matplotlib.pyplot as plt

def plot_bar_chart(x_vals, y_vals, x_label, y_label, title, disp_bar_vals=False, size=None):
    """Displayes a bar chart of the given x and y data.

    Args:
        x(list): list of x values
        y(list): list of y values
        x_label(str): label for the x axis
        y_label(str): label for the y axis
        title(str): title of the chart
        disp_bar_vals(bool): whether the bars should display their values
        size(tuple of int): size of graph
    """
    plt.figure(figsize=size)
    plt.bar(x_vals, y_vals, color=(0, 0.5, 1.0), edgecolor="black")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=28, ha="right")
    if disp_bar_vals:
        for index, value in enumerate(y_vals):
            plt.text(index, value + 15, str(value), ha="center", va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_pie_chart(x_vals, y_vals, title):
    """Displayes a pie chart of the given x and y data.

    Args:
        x(list): list of x values
        y(list): list of y values
        title(str): title of the chart
    """
    plt.figure(figsize=(6, 6))
    plt.pie(y_vals, labels=x_vals, autopct="%1.1f%%")
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

def scatter_plot(x_vals, y_vals, size, labels, title):
    """Displayes a scatter plot of the given x and y data.

    Args:
        x_vals(list): list of x values
        y_vals(list): list of y values
        size(int): size of the markers
        x_label(str): label for the x axis
        y_label(str): label for the y axis
        title(str): title of the chart
    """
    # start by calculating values for the best fit line
    slope, intercept = compute_slope_intercept(x_vals, y_vals)
    line_y_values = [value * slope + intercept for value in x_vals]
    # use function to calculate correlation and covariance
    corr, cov = compute_correlation_and_covariance(x_vals, y_vals)
    text = "corr: " + str(round(corr, 2)) + "; cov: " + \
        str(round(cov, 2))
    # now plot the chart
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, marker=".", s=size, c="blue")
    # The annotate code is from Stack Overflow as I did not know how else to
    # include the box with the correlation and covariance values.
    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), color='red', fontsize=10,
        xycoords='axes fraction', textcoords='offset points',
        bbox=dict(facecolor='white', alpha=0.8, ec='red'),
        horizontalalignment='right', verticalalignment='top')
    plt.plot(x_vals, line_y_values, 'r')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.show()

def plot_histogram(data, x_label, y_label, title):
    """Displayes a histogram of the given data.

    Args:
        data(list): list of data
        x_label(str): label for the x axis
        y_label(str): label for the y axis
        title(str): title of the chart
    """
    plt.figure()
    plt.hist(data, color=(0, 0.5, 1.0)) # default bins is 10 so no need to specify
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def compute_slope_intercept(x_vals, y_vals):
    """Fits a simple univariate line y = mx + b to the provided x y data.
    Follows the least squares approach for simple linear regression.

    Args:
        x_vals(list of numeric vals): The list of x values
        y_vals(list of numeric vals): The list of y values

    Returns:
        slope(float): The slope of the line fit to x and y
        intercept(float): The intercept of the line fit to x and y
    """
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    slope = sum([(x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(len(x_vals))]) \
        / sum([(x_vals[i] - x_mean) ** 2 for i in range(len(x_vals))])
    intercept = y_mean - slope * x_mean
    return slope, intercept

def compute_correlation_and_covariance(x_vals, y_vals):
    """Computes the correlation coefficient and covariance
    value for the given data.

    Args:
        x_vals(list of numeric vals): The list of x values
        y_vals(list of numeric vals): The list of y values

    Returns:
        corr(float): The correlation coeffient
        cov(float): The covariance value.
    """
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    corr = sum([(x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(len(x_vals))]) \
        / (sum([(x_vals[i] - x_mean) ** 2 for i in range(len(x_vals))]) \
                * sum([(y_vals[i] - y_mean) ** 2 for i in range(len(y_vals))])) ** (1/2)
    cov = sum([(x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(len(x_vals))]) \
        / len(x_vals)
    return corr, cov
