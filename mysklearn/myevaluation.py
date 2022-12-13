"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains various algorithms for evaluating classifiers.
"""
import numpy as np # use numpy's random number generation

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # copy samples
    X_train = X[:]
    y_train = y[:]
    n = len(X)

    # shuffle list if specified to do so
    if shuffle:
        shuffle_data(X_train, y_train, random_state)

    # select index for where the samples will be split into train and test sets
    # if the value is greater than 1.0 (100%), assume it is supposed to be the number of
    # test instances. Otherwise, assume it is a percentage of the sample sizes.
    if float(test_size) >= 1.0:
        split_index = int(n - test_size)
    else:
        split_index = int((1.0 - test_size) * n)
    # return the train and test sets
    return X_train[0:split_index], X_train[split_index:], \
        y_train[0:split_index], y_train[split_index:]

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # copy sample indexes
    X_train = [X.index(sample) for sample in X]
    folded_samples = []
    folds = []

    # creates the number of folds specified by n_splits
    for _ in range(0, n_splits):
        folded_samples.append([])

    # shuffle list if specified to do so
    if shuffle:
        shuffle_data(X_train, random_state=random_state)

    # split X_train into n folds
    for index, sample in enumerate(X_train):
        folded_samples[index % n_splits].append(sample)

    # for each fold, set the fold as the test set and all other folds as training set
    # then append that tuple to the list of folds
    for samples_index, sample_fold in enumerate(folded_samples):
        fold = [[]]
        fold.append(sample_fold)
        for fold_index in range(0, n_splits):
            for sample in folded_samples[fold_index]:
                if samples_index != fold_index:
                    fold[0].append(sample)
        folds.append(tuple(fold))
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # copy sample indexes
    X_train = list(range(len(X)))
    y_train = y[:]
    folded_samples = []
    folds = []
    unique_y_vals = []

    # find unique y_values for later use
    for y_val in y_train:
        if y_val not in unique_y_vals:
            unique_y_vals.append(y_val)
    unique_y_vals.sort()

    # creates the number of folds specified by n_splits
    for _ in range(0, n_splits):
        folded_samples.append([])

    # shuffle list if specified to do so
    if shuffle:
        shuffle_data(X_train, y_train, random_state)

    # split X_train into n folds
    # distributes values so each fold has about the same percentage of each class label
    for y_val in unique_y_vals:
        count = 0
        for index, sample in enumerate(X_train):
            if y_train[index] == y_val:
                folded_samples[count % n_splits].append(sample)
                count += 1

    # for each fold, set the fold as the test set and all other folds as training set
    # then append that tuple to the list of folds
    for samples_index, sample_fold in enumerate(folded_samples):
        fold = [[]]
        fold.append(sample_fold)
        for fold_index in range(0, n_splits):
            for sample in folded_samples[fold_index]:
                if samples_index != fold_index:
                    fold[0].append(sample)
        folds.append(tuple(fold))
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    # initialize variables
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    selected_row_indexes = []
    n = len(X)

    # if an n value is given, use it instead of len(X)
    if n_samples is not None:
        n = n_samples

    # shuffle list if specified to do so
    if random_state is not None:
        np.random.seed(random_state)

    # randomly choose an element from the list and add it to the samples list, and add its index to the
    # selected indexes list. repeat this n times (n is number of elements in the data)
    for _ in range(0, n):
        index = np.random.randint(0, n)
        X_sample.append(X[index])
        # only use y values if a list of y values is given
        if y is not None:
            y_sample.append(y[index])
        if index not in selected_row_indexes:
            selected_row_indexes.append(index)

    # add all the unused samples to the out of bag lists
    for index, sample in enumerate(X):
        if index not in selected_row_indexes:
            X_out_of_bag.append(sample)
            if y is not None:
                y_out_of_bag.append(y[index])
    # return the saple and out of bag lists
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    num_labels = len(labels)

    # creates a matrix of 0's with shape of (num_lables, num_lables)
    matrix = []
    for _ in range(num_labels):
        matrix.append([0] * num_labels)

    # fill matrix with counts of correctly and incorrectly classified labels
    for index, true_val in enumerate(y_true):
        # get index of the label in labels
        true_index = labels.index(true_val)
        if true_val == y_pred[index]:
            matrix[true_index][true_index] += 1
        else:
            pred_index = labels.index(y_pred[index])
            matrix[true_index][pred_index] += 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = 0
    # add 1 to correct_count for each pair of matching y_pred and y_true values
    for index, true_val in enumerate(y_true):
        if true_val == y_pred[index]:
            correct_count += 1
    # return the fraction of the correctly classified samples if normalize is true
    if normalize:
        return float(correct_count) / len(y_true)
    return correct_count

def shuffle_data(X, y=None, random_state=None):
    """Randomly shuffles the X and y (if given) data.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            Default is None (if the calling code only wants to sample X)
        random_state(int): integer used for seeding a random number generator for reproducible results.

    Notes:
        Performs inplace shuffling on the given lists.
    """
    n = len(X)
    if random_state is not None:
        np.random.seed(random_state)
    for i in range(n):
        # pick an index to swap
        j = np.random.randint(0, n) # random int in [0,n)
        X[i], X[j] = X[j], X[i]
        if y is not None:
            y[i], y[j] = y[j], y[i]

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    true_positives = 0
    false_positives = 0
    possible_labels = labels
    positive_label = pos_label

    if possible_labels is None:
        possible_labels = list(np.unique(y_true))

    if positive_label is None:
        positive_label = possible_labels[0]
    #print(positive_label)

    # calculate TP and FP values
    for index, true_val in enumerate(y_true):
        pred_val = y_pred[index]
        if pred_val == positive_label:
            # if true_val and pred_val are positive: true postive
            if true_val == positive_label:
                true_positives += 1
            # true_val is not positive: false positive
            else:
                false_positives += 1

    # if there were no TP and FP values, return 0 (this is to avoid dividing by 0)
    if true_positives + false_positives == 0:
        return 0

    # otherwise return the precision
    return float(true_positives) / (true_positives + false_positives)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    true_positives = 0
    false_negatives = 0
    possible_labels = labels
    positive_label = pos_label

    if possible_labels is None:
        possible_labels = list(np.unique(y_true))

    if positive_label is None:
        positive_label = possible_labels[0]
    #print(positive_label)

    # calculate TP and FN values
    for index, true_val in enumerate(y_true):
        pred_val = y_pred[index]
        if true_val == positive_label:
            # if true_val and pred_val are positive: true postive
            if pred_val == positive_label:
                true_positives += 1
            # pred_val is not positive: false negative
            else:
                false_negatives += 1

    # if there were no TP and FN values, return 0 (this is to avoid dividing by 0)
    if true_positives + false_negatives == 0:
        return 0

    # otherwise return the recall
    return float(true_positives) / (true_positives + false_negatives)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # calculate precision and recall
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # return 0 if precision or recall are equal to 0
    if precision == 0 or recall == 0:
        return 0

    # otherwise return the F1 score
    return 2 * (precision * recall) / (precision + recall)
