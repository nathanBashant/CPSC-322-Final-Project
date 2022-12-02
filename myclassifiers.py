"""
Reid Plowman, Nathan Bashant
CPSC 322-02, Fall 2022
Partner Project
12/1/2022

Description: This module contains classes different classifiers.
"""
import operator
import math
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
        neighbor_indices = self.kneighbors(X_test)[1]
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
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None

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
        # basic approach (uses recursion!!):
        #print("available attributes:", available_attributes)
        #print(current_instances)

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