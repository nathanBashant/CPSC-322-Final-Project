"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/12/2022

Description: This module contains test functions for methods in the classes
contained in the mysklearn.myclassifiers module.
"""
import numpy as np

from mysklearn.myclassifiers import MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyDecisionTreeClassifier

# note: order is actual/received value, expected/solution
def test_kneighbors_classifier_kneighbors():
    """Test function
    """
    # testing against in-class example 1
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    knn_clf = MyKNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([[0.33, 1]])
    assert np.allclose(distances, [[0.67, 1.0, 1.053]])
    assert np.allclose(indexes, [[0, 2, 3]])
    # np.allclose is not needed for the int values, but it made the comparisons easier

    # testing against in-class example 2
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn_clf = MyKNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([[2, 3]])
    assert np.allclose(distances, [[1.414, 1.414, 2.0]])
    assert np.allclose(indexes, [[0, 4, 6]])

    # testing against Bramer exercise
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
        "-", "-", "+", "+", "+", "-", "+"]
    knn_clf = MyKNeighborsClassifier(5)
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([[2, 3]])
    assert np.allclose(distances, [[3.511, 4.401, 5.135, 9.617, 10.733]])
    assert np.allclose(indexes, [[0, 2, 1, 5, 4]])

def test_kneighbors_classifier_predict():
    """Test function
    """
    # testing against in-class example 1
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    knn_clf = MyKNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict([[0.33, 1]])
    assert y_pred[0] == "good"

    # testing against in-class example 2
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn_clf = MyKNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict([[2, 3]])
    assert y_pred[0] == "yes"

    # testing against Bramer exercise
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
        "-", "-", "+", "+", "+", "-", "+"]
    knn_clf = MyKNeighborsClassifier(5)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict([[2, 3]])
    assert y_pred[0] == "-"

def test_dummy_classifier_fit():
    """Test function
    """
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100,
        replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == "no"

    # made up y values with root beer being the most frequent label
    y_train = list(np.random.choice(["pepsi", "root beer", "sprite"],
        100, replace=True, p=[0.3, 0.55, 0.15]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    assert dummy_clf.most_common_label == "root beer"

def test_dummy_classifier_predict():
    """Test function
    """
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict([[2, 3]])
    assert y_pred[0] == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100,
        replace=True, p=[0.2, 0.6, 0.2]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict([[2, 3]])
    assert y_pred[0] == "no"

    # made up y values with root beer being the most frequent label
    y_train = list(np.random.choice(["pepsi", "root beer", "sprite"],
        100, replace=True, p=[0.3, 0.55, 0.15]))
    dummy_clf = MyDummyClassifier()
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict([[3, 6], [2, 4], [4, 4]])
    for predicted_value in y_pred:
        assert predicted_value == "root beer"

def test_decision_tree_classifier_fit():
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

    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior",
                    ["Attribute", "att3",
                        ["Value", "no",
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    treeclf = MyDecisionTreeClassifier()
    treeclf.fit(X_train_interview, y_train_interview)
    assert treeclf.tree == tree_interview

    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone = \
            ["Attribute", "att0",
                ["Value", 1,
                    ["Attribute", "att1",
                        ["Value", 1,
                            ["Leaf", "yes", 1, 5]
                        ],
                        ["Value", 2,
                            ["Attribute", "att2",
                                ["Value", "excellent",
                                    ["Leaf", "yes", 1, 2]
                                ],
                                ["Value", "fair",
                                    ["Leaf", "no", 1, 2]
                                ]
                            ]
                        ],
                        ["Value", 3,
                            ["Leaf", "no", 2, 5]
                        ],
                    ]
                ],
                ["Value", 2,
                    ["Attribute", "att2",
                        ["Value", "excellent",
                            ["Leaf", "no", 4, 10]
                        ],
                        ["Value", "fair",
                            ["Leaf", "yes", 6, 10]
                        ]
                    ]
                ]
            ]
    treeclf = MyDecisionTreeClassifier()
    treeclf.fit(X_train_iphone, y_train_iphone)
    assert treeclf.tree == tree_iphone

def test_decision_tree_classifier_predict():
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

    treeclf = MyDecisionTreeClassifier()
    treeclf.fit(X_train_interview, y_train_interview)

    interview_test1 = ["Junior", "Java", "yes", "no"]
    interview_test2 = ["Junior", "Java", "yes", "yes"]

    assert treeclf.predict([interview_test1, interview_test2]) == ["True", "False"]

    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    treeclf = MyDecisionTreeClassifier()
    treeclf.fit(X_train_iphone, y_train_iphone)

    iphone_test1 = [2, 2, "fair"]
    iphone_test2 = [1, 1, "excellent"]

    assert treeclf.predict([iphone_test1, iphone_test2]) == ["yes", "yes"]
