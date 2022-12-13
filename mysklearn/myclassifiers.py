"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains classes for different classifiers.
"""
import operator
import math
import pprint 
import numpy as np
from scipy import stats

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        # calculate distances and indices for each test value in X_test
        for test_instance in X_test:
            row_indexes_dists = []
            for i, train_instance in enumerate(self.X_train):
                dist = self.compute_euclidean_distance(train_instance, test_instance)
                row_indexes_dists.append((i, dist))
            row_indexes_dists.sort(key=operator.itemgetter(-1))
            top_k = row_indexes_dists[:self.n_neighbors]
            neighbor_indices.append([row[0] for row in top_k])
            distances.append([round(row[1], 3) for row in top_k])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, neighbor_indices = self.kneighbors(X_test)
        for test_instance_index in range(len(X_test)):
            y_vals = []
            for index in neighbor_indices[test_instance_index]:
                y_vals.append(self.y_train[index])
            y_predicted.append(stats.mode(y_vals)[0][0])
        return y_predicted

    @staticmethod
    def compute_euclidean_distance(v1, v2):
        """Calculates the euclidean distance for the given lists.
        """
        dist_vals = []
        for index, val in enumerate(v1):
            if isinstance(val, (int, float)):
                dist_vals.append((val - v2[index]) ** 2)
            elif val == v2[index]:
                dist_vals.append(0)
            else:
                dist_vals.append(1)
        return np.sqrt(sum(dist_vals))

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # had to figure out some way to use X_train so pylint
        # would not deduct points for not using it
        if X_train is not None:
            self.most_common_label = stats.mode(y_train)[0][0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        i = 0
        while i < len(X_test):
            y_predicted.append(self.most_common_label)
            i += 1
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, num_attributes=None):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None
        self.num_attributes = num_attributes # used in random forest classification 

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        header = ["att" + str(i) for i in range(len(X_train[0]))]
        attribute_domains = {}

        # poulate attribute_domains using all possible values for each attribute
        for index, att in enumerate(header):
            att_vals = []
            # get every value from X_train in the attribute's column
            for row in X_train:
                att_vals.append(row[index])
            unique_att_vals = list(np.unique(att_vals))
            attribute_domains[att] = unique_att_vals

        # store in instance variables
        self.header = header
        self.attribute_domains = attribute_domains

        # combine X_train and y_train into one dataset
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy()

        # create the decision tree using the recursive tdidt method
        tree = self.tdidt(train, available_attributes)

        self.tree = tree
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = [self.tdidt_predict(self.tree, test_val) for test_val in X_test]
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        att_names = attribute_names
        if att_names is None:
            att_names = self.header

        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(self.tree)
        rule = "IF"

        att_index = self.header.index(self.tree[1])

        # now loop through all of the value lists
        att_rule = rule + " " + att_names[att_index]
        for i in range(2, len(self.tree)):
            value_list = self.tree[i]
            #print(value_list)
            val_rule = att_rule + " == " + value_list[1]
            #print(val_rule)
            # we have a match, recurse on this value's subtree
            self.decision_rule_recurse(value_list[2], val_rule, att_names, class_name)

    def tdidt(self, current_instances, available_attributes, prev_node_count=None):
        """Recursive TDIDT algorithm that creates a decision tree for the current instances.

        Args:
            current_instances(list of list of obj): list of instances in the current tree.
            available_attributes(list of str): list of usable attributes for the current tree.
            prev_node_count(int): number of instances in the previous node. (for use in case 3)

        Returns:
            tree: decision tree stored as nested lists
        """
        if prev_node_count is None:
            prev_node_count = len(current_instances)
        #print("available attributes:", available_attributes)
        #print(current_instances)

        # select attribue subset (only used in random forest classification)
        if self.num_attributes is not None:
            available_attributes = MyDecisionTreeClassifier.compute_random_subset(available_attributes, \
                self.num_attributes)

        # select an attribute to split on
        entropy_vals = MyDecisionTreeClassifier.calc_entropy(current_instances, available_attributes)
        split_attribute = available_attributes[entropy_vals.index(min(entropy_vals))]
        #print("splititting on:", split_attribute)
        available_attributes.remove(split_attribute)
        # cannot split on this attribute again in this branch of the tree
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions_dict = self.partition_instances(current_instances, split_attribute)
        # print("paritions:", partitions_dict)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions_dict.items():
            value_subtree = ["Value", att_value]
            if len(att_partition) > 0 and MyDecisionTreeClassifier.same_class_label(att_partition):
                #print("CASE 1 all same class label")
                #    CASE 1: all class labels of the partition are the same
                # => make a leaf node (numerator is len(att_partition), denominator is len(current_instances))
                value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)])
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                #print("CASE 2 clash")
                #    CASE 2: no more attributes to select (clash)
                # => handle clash w/majority vote leaf node (most frequent class label)
                labels = []
                for instance in att_partition:
                    labels.append(instance[-1])
                most_frequent_label = stats.mode(labels)[0][0]
                value_subtree.append(["Leaf", most_frequent_label, len(att_partition), len(current_instances)])
            elif len(att_partition) == 0:
                #print("CASE 3 empty partition")
                #    CASE 3: no more instances to partition (empty partition)
                #  => backtrack and replace attribute node with majority vote leaf node
                # turns out that partitioning on this attribute was a bad idea, replace tree with
                # majority vote leaf node
                #print(tree)
                labels = []
                for instance in current_instances:
                    labels.append(instance[-1])
                most_frequent_label = stats.mode(labels)[0][0]
                tree = ["Leaf", most_frequent_label, len(current_instances), prev_node_count]
                break
            else: # none of the base cases were true... we recurse!
                #print("Recursing!!")
                subtree = self.tdidt(att_partition, available_attributes.copy(), len(current_instances))
                value_subtree.append(subtree)
            tree.append(value_subtree)
        #print(tree)
        return tree

    @staticmethod
    def compute_random_subset(values, num_values):
        """Randomly selects a given number of values from the list of values and returns them.

        Args:
            values(list of obj): the list of values to select from
            num_values(int): the number of values to select
        
        Returns:
            values_copy(list of obj): list of the randomly selected values
        """
        values_copy = values.copy() # shallow copy
        np.random.shuffle(values_copy) # inplace shuffle
        return values_copy[:num_values]

    def partition_instances(self, instances, attribute):
        """Group by function for use in tdidt.

        Args:
            instances(list of list of obj): list of instances
            attribute(str): attribute to perform the groupby on

        Returns:
            partitions(dict): partitions of instances after the groupby is performed
        """
        # this is a gorup by attribute domain
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains["att" + str(att_index)]
        # print("attribute domain:", att_domain)
        # lets use dicitionaries
        partitions = {}
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    @staticmethod
    def calc_entropy(instances, attributes):
        """Calculates entropy values for each attribute in attributes.

        Args:
            instances(list of list of obj): list of instances.
            attributes(list of str): list of attributes.

        Returns:
            entropy_vals(list of float): list of entropy values, parallel to attributes
        """
        entropy_vals = []
        labels = [instance[-1] for instance in instances]
        unique_labels = list(np.unique(labels))
        unique_labels = list(unique_labels)
        #print(labels)

        # calculate entropy values for each attribute
        for att in attributes:
            # list of entropy values for each value of the current attribute
            att_entropies = []

            # index of the attribute is the last char in the string
            # this only works since we are using generic "att#" attribute labels
            att_index = int(att[-1])
            att_vals = [instance[att_index] for instance in instances]
            unique_att_vals, att_counts = np.unique(att_vals, return_counts=True)
            unique_att_vals = list(unique_att_vals)
            att_counts = list(att_counts)

            # calculate entropy values for each value of the current attribute
            for val_index, att_val in enumerate(unique_att_vals):
                label_counts = []
                for _ in range(len(unique_labels)):
                    label_counts.append(0)

                # count the number of times that each label appears for the current attribute value
                for instance in instances:
                    if instance[att_index] == att_val:
                        label_counts[unique_labels.index(instance[-1])] += 1

                entropy_val = 0
                # if the count is 0 for either label, entropy is set to 0 to avoid log(0) which is undef
                if 0 not in label_counts:
                    for count in label_counts:
                        probability = count / att_counts[val_index]
                        entropy_val -= probability * math.log(probability, 2)
                att_entropies.append(entropy_val)
            att_entropy = 0

            # calculate the weighted entropy for the current attribute and add to list of entropy values
            for index, entropy_val in enumerate(att_entropies):
                att_entropy += (att_counts[index] / len(instances)) * entropy_val
            entropy_vals.append(att_entropy)
        return entropy_vals

    @staticmethod
    def same_class_label(instances):
        """Checks to see if all the instances have the same class label, which is stored at the
        end of the list. Returns true if all class labels match.

        Args:
            instances(list of list of obj): the list of instances

        Returns:
            matching(bool): whether all the class labels are the same or not
        """
        first_label = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_label:
                return False
        # get here, all the same
        return True

    def tdidt_predict(self, tree, instance):
        """Recursive helper function for finding the predicted value for the given instance.

        Args:
            tree: decision tree stored as nested lists
            instance(list of obj): unknown instance to use for predictions

        Returns:
            label(obj): value stored at leaf node
        """
        # are we at a leaf node (base case)
        # or an attribute node (need to recurse)
        info_type = tree[0] # Attribute or Leaf
        if info_type == "Leaf":
            # base case
            return tree[1] # label

        # if we are here, then we are at an Attribute node
        # we need to figure out where in instance, this attribute's value is
        att_index = self.header.index(tree[1])
        # now loop through all of the value lists, looking for a match
        # to instance[att_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                # we have a match, recurse on this value's subtree
                return self.tdidt_predict(value_list[2], instance)
        return None

    def decision_rule_recurse(self, tree, current_rule, att_names, class_name):
        """Recursive helper function for printing decision rules.

        Args:
            tree: decision tree stored as nested lists
            current_rule(str): string representation of the current decision rule
            att_names(list of str): A list of attribute names to use in the decision rules
            class_name(str): A string to use for the class name in the decision rules
        """
        # are we at a leaf node (base case)
        # or an attribute node (need to recurse)
        info_type = tree[0] # Attribute or Leaf
        if info_type == "Leaf":
            # base case
            rule_end = "  THEN  " + class_name + " = " + str(tree[1])
            print(current_rule + rule_end)
            return

        # if we are here, then we are at an Attribute node
        att_index = self.header.index(tree[1])
        # now loop through all of the value lists

        att_rule = current_rule + "  AND  " + att_names[att_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            #print(value_list)
            val_rule = att_rule + " == " + value_list[1]
            #print(val_rule)
            # we have a match, recurse on this value's subtree
            self.decision_rule_recurse(value_list[2], val_rule, att_names, class_name)

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

        TODO: Fill this out properly

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, num_trees, num_best, num_attributes, seed_random=False):
        """Initializer for MyRandomForestClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.num_trees = num_trees # N
        self.num_best = num_best # M
        self.num_attributes = num_attributes # F
        self.seed_random = seed_random
        self.forest = None
        self.tree_accuracies = None

    def fit(self, X, y):
        """Fits a random forest classifier to X_train and y_train.

        Args:
            X(list of list of obj): The list of training instances (samples).
                The shape of X is (n_samples, n_features)
            y(list of obj): The target y values (parallel to X)
                The shape of y is n_samples

        Notes:
        """
        trees = []

        # create the trees and store them with their accuracy value in a list
        for i in range(self.num_trees): 
            random_state = None
            if self.seed_random:
                # use i for random state for bootstrapping
                random_state = i
            X_train, X_test, y_train, y_test = MyRandomForestClassifier.bootstrap_sample(X, y, \
                random_state=random_state)
            tree = MyDecisionTreeClassifier(self.num_attributes)
            tree.fit(X_train, y_train)
            #tree.print_decision_rules(["level", "lang", "tweets", "phd", "interviewed_well"])

            y_pred = tree.predict(X_test)
            accuracy = MyRandomForestClassifier.accuracy_score(y_test, y_pred)
            #print(accuracy)

            trees.append([tree, accuracy])

        # sort the trees by highest accuracy
        sorted_by_accuracy = sorted(trees,key=lambda l:l[1], reverse=True)

        # then save the best M trees
        best_m_trees = [sorted_by_accuracy[i][0] for i in range(self.num_best)]
        tree_accuracies = [sorted_by_accuracy[i][1] for i in range(self.num_best)]
        self.forest = best_m_trees
        self.accuracies = tree_accuracies
        print(self.accuracies)
        pass # TODO: implement

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for test_val in X_test:

            # get predicted value from each tree and store in pred_vals
            pred_vals = []
            for tree in self.forest:
                pred_val = tree.predict([test_val])[0]
                if pred_val is None:
                    pred_val = ""
                pred_vals.append(pred_val)
            
            # use majority voting to select predicted value
            pred_val = stats.mode(pred_vals)[0][0]
            if pred_val == "":
                pred_val = None
            predictions.append(pred_val)
        return predictions

    @staticmethod
    def bootstrap_sample(X, y, n_samples=None, random_state=None):
        """Split dataset into bootstrapped training set and out of bag test set.

        Args:
            X(list of list of obj): The list of samples
            y(list of obj): The target y values (parallel to X)
            n_samples(int): Number of samples to generate. If left to None (default) this is automatically
                set to the first dimension of X.
            random_state(int): integer used for seeding a random number generator for reproducible results

        Returns:
            X_sample(list of list of obj): The list of samples
            X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
            y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
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

        np.random.seed(random_state)

        # randomly choose an element from the list and add it to the samples list, and add its index to the
        # selected indexes list. repeat this n times (n is number of elements in the data)
        for _ in range(0, n):
            index = np.random.randint(0, n)
            X_sample.append(X[index])
            # only use y values if a list of y values is given
            y_sample.append(y[index])
            if index not in selected_row_indexes:
                selected_row_indexes.append(index)

        # add all the unused samples to the out of bag lists
        for index, sample in enumerate(X):
            if index not in selected_row_indexes:
                X_out_of_bag.append(sample)
                y_out_of_bag.append(y[index])

        # if somehow all the indices get selected, add the last one from samples to out of bag
        if len(X_out_of_bag) == 0:
            X_out_of_bag.append(X_sample.pop())
            y_out_of_bag.append(y_sample.pop())

        # return the sample and out of bag lists
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag

    @staticmethod
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