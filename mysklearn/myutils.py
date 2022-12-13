"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains utility functions for use in the Jupyter Notebook.
"""
import numpy as np
from tabulate import tabulate

from mysklearn import myevaluation

def discretizer(var):
    if var >= 80:
        return "excellent"
    if var >= 50:
        return "fair"
    return "low"

def get_frequencies_col(col_name):
    col = col_name.copy()
    col.sort() 
    
    values = [] 
    counts = []
    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts

def get_columns_of_table(table, col_names):
    """Returns a 2D list with the data in the specified columns.

    Args:
        col_names(list of str): column names to create a new 2D list from

    Returns:
        data(list of list): new 2D list containing only the data from the specified columns.
    """
    col_indexes = [table.column_names.index(col_name) for col_name in col_names]
    data = []
    for row in table.data:
        row_data = []
        for col_index in col_indexes:
            row_data.append(row[col_index])
        data.append(row_data)
    return data

def calc_classifier_performance(X, y, clf, clf_name, folds, pos_label=None, class_name=None):
    """Calculates and prints the accuracy, error rate, precision, recall,
        and F1 score of the classifier for the given X and y values. Uses stratified k-fold
        cross validation.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
        clf: classifier to use to make predictions
        clf_name(str): name of the classifier to use for printing
        folds(list of 2-item tuples): folds returned by stratified_kfold_split()

    Notes:
        clf can be any of the classifiers in the myclassifiers module since
        their fit() and predict() methods have the same parameters and returns
    """
    X_values = X.copy()
    y_values = y.copy()
    unique_y_vals = list(np.unique(y_values))
    true_vals = []
    pred_vals = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for fold in folds:
        # unpack tuple
        train_indexes, test_indexes = fold

        # create testing and training sets
        X_train = [X_values[index] for index in train_indexes]
        X_test = [X_values[index] for index in test_indexes]
        y_train = [y_values[index] for index in train_indexes]
        y_test = [y_values[index] for index in test_indexes]
        true_vals.extend(y_test)

        # fit the data to the calssifier and predict the results of the test values
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred_vals.extend(y_pred)

        # calculate accuracy, precision, recall, and f1 score for current fold,
        # then append each to their respective total lists
        accuracy_list.append(myevaluation.accuracy_score(y_test, y_pred))
        precision_list.append(myevaluation.binary_precision_score(y_test, y_pred, \
            unique_y_vals, pos_label))
        recall_list.append(myevaluation.binary_recall_score(y_test, y_pred, \
            unique_y_vals, pos_label))
        f1_list.append(myevaluation.binary_f1_score(y_test, y_pred, \
            unique_y_vals, pos_label))
    # create confusion matrix from overall true and predicted values
    matrix = myevaluation.confusion_matrix(true_vals, pred_vals, unique_y_vals)
    matrix = add_stats_to_matrix(matrix, class_name, unique_y_vals)

    # print performance metrics
    print(clf_name, "Classifier Performance Metrics:")
    print("Accuracy: " + str(round(np.mean(accuracy_list), 4)))
    print("Error rate:", round((1 - np.mean(accuracy_list)), 4))
    print("Precision:", round(np.mean(precision_list), 4))
    print("Recall:", round(np.mean(recall_list), 4))
    print("F1 Score:", round(np.mean(f1_list), 4))
    print("Confusion Matrix:")
    print(tabulate(matrix, headers="firstrow"))

def add_stats_to_matrix(matrix, class_name, labels):
    """Helper function that adds labels and headers to the matrix. Also adds the totals
    and recognition (as a %) to each row.

    Args:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
        class_name(str): name for the class labels (goes in top left of matrix)
        labels(str): list of labels for the values in the matrix

    Returns:
        stats_matrix(list of list of obj): Confusion matrix with header, row labels,
        totals of each row, and recognition (as a %) for each row
    """
    stats_matrix = []
    header = []

    # add values to the header
    header.append(class_name)
    header.extend(labels)
    header.extend(["Total", "Recognition (%)"])
    stats_matrix.append(header)

    # add values to each row
    for index, row in enumerate(matrix):
        new_row = []
        # start each row with the row's label from labels
        new_row.append(labels[index])
        # then add the original row of data
        new_row.extend(row)
        # then add the sum of values in the row
        new_row.append(sum(row))
        # then add the recognition (as a %) of the row
        # calculated by dividing that row's TP count by the sum of that row
        # and multiplying by 100. if TP = 0, recognition (%) = 0
        if row[index] == 0:
            new_row.append(0)
        else:
            new_row.append(round(row[index] / sum(row), 4) * 100)
        # add new row to new matrix
        stats_matrix.append(new_row)
    return stats_matrix
