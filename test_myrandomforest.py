"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/12/2022

Description: This module contains test functions for methods in the MyRandomForestClassifier class.
"""
import numpy as np

from mysklearn import myevaluation
from mysklearn.myclassifiers import MyRandomForestClassifier

def test_random_forest_predict():
    """Test function
    """
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    remainder_indices, test_indices = myevaluation.stratified_kfold_split(X_train_interview, \
        y_train_interview, 3, shuffle=True)[0]
    print(remainder_indices, test_indices)

    X_remainder = [X_train_interview[index] for index in remainder_indices]
    y_remainder = [y_train_interview[index] for index in remainder_indices]
    X_test = [X_train_interview[index] for index in test_indices]
    y_test = [y_train_interview[index] for index in test_indices]

    forest_clf = MyRandomForestClassifier(20, 7, 2)
    forest_clf.fit(X_remainder, y_remainder)

    y_pred = forest_clf.predict(X_test)
    correct_count = 0
    for index, val in enumerate(y_pred):
        if y_test[index] == val:
            correct_count += 1

    accuracy = correct_count / len(y_test)
    print("Pred values:", y_pred)
    print("True values:", y_test)
    print("Accuracy:", str(accuracy * 100) + "%")
    assert correct_count / len(y_test) >= 0.8 # with complete randomness, given the small dataset,
    # aiming for 100% accuracy would take a while quite a few attempts of rerunning the code

def test_seeded_random_forest_predict():
    """Test function
    """
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    remainder_indices, test_indices = myevaluation.stratified_kfold_split(X_train_interview, \
        y_train_interview, 3, 12, True)[0]
    #print(remainder_indices, test_indices)

    X_remainder = [X_train_interview[index] for index in remainder_indices]
    y_remainder = [y_train_interview[index] for index in remainder_indices]
    X_test = [X_train_interview[index] for index in test_indices]
    y_test = [y_train_interview[index] for index in test_indices]

    forest_clf = MyRandomForestClassifier(20, 7, 2, True)
    forest_clf.fit(X_remainder, y_remainder)

    y_pred = forest_clf.predict(X_test)
    #print("pred vals:", y_pred)
    #print("true vals:", y_test)
    assert y_pred == y_test
