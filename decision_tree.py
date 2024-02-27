"""
This file will define the decision tree class which is used for the random forrest clsssifier

written by: Tshepo Maredi
"""

import numpy as np


class Node:
    def __init__(self, attribute=None, left_child=None, right_child=None, node_value=None, threshold=None):
        self.attribute = attribute
        self.left_child = left_child
        self.right_child = right_child
        self.node_value = node_value
        self.threshold = threshold

    def is_leaf_node(self):
        if self.node_value is not None:
            return True
        else:
            return False


class DecisionTree:
    def __init__(self, minimum_samples_split=2, maximum_tree_depth=100, num_of_attributes=None):
        """
        initialization of the decision tree
        :param minimum_samples_split:(int) minimum amount of samples for a split to happen
        :param maximum_tree_depth: (int)
        :param num_of_attributes: (int) number of attributes/features
        """
        self.minimum_samples_splits = minimum_samples_split
        self.maximum_tree_depth = maximum_tree_depth
        self.num_of_attributes = num_of_attributes
        self.root_node = None


    def fit(self, X, y):
        """
        fit function will set the number of attributes and expand the tree. This is creating the tree
        :param X: 2D [] of shape (# of samples, # of attributes)
        :param y: [] of shape (# samples, _)
        :return: None. Will set the root node for the full tree
        """
        if not self.num_of_attributes:
            self.num_of_attributes = X.shape[1]
        else:
            self.num_of_attributes = min(X.shape[1], self.num_of_attributes)

        self.root_node = self.expand_tree(X, y)

    def expand_tree(self, X, y, max_tree_depth=0):
        """
        will grow the decision tree to the maximum tree depth defined
        :param X: 2D [] of shape (# of samples, # of attributes)
        :param y: [] of shape (# samples, _)
        :param max_tree_depth: (int)
        :return: root Node
        """
        y = np.array(y)
        number_of_samples, number_of_attributes = X.shape
        num_labels = len(np.unique(y))

        # check the stop criteria
        if max_tree_depth >= self.maximum_tree_depth or num_labels == 1 or number_of_samples < self.minimum_samples_splits:
            leaf_node_value = self.find_most_common_label(y) # plurailiy value = most common label/output value among a set of examples
            return Node(node_value=leaf_node_value)

        # TODO:
        attribute_indices = np.random.choice(number_of_attributes, self.num_of_attributes, replace=False)

        # find the best split attributes
        best_split_threshold, best_split_attribute = self.find_best_split_attributes(X, y, attribute_indices)
        # create the child nodes
        left_indices, right_indices = self.perform_split(X[:, best_split_attribute], best_split_threshold)

        left = self.expand_tree(X[left_indices, :], y[left_indices], max_tree_depth+1)
        right = self.expand_tree(X[right_indices, :], y[right_indices], max_tree_depth+1)
        return Node(attribute=best_split_attribute, threshold=best_split_threshold, left_child=left, right_child=right)





    def perform_split(self, X_feat_column, split_threshold):
        """

        :param X_feat_column:
        :param split_threshold:
        :return:
        """
        # TODO: remove argwhere
        left_indices = np.argwhere(X_feat_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_feat_column > split_threshold).flatten()
        return left_indices, right_indices


    def find_best_split_attributes(self, X, y, attr_indices):
        """
        finds the best split attributes
        :param X: 2D [] of shape (# of samples, # of attributes)
        :param y: [] of shape (# samples, _)
        :param attr_indices:
        :return:
        """
        highest_gain_value = -1
        split_index, split_gain_threshold = None, None

        for attribute_index in attr_indices:
            X_feat_column = X[:, attribute_index]
            split_thresholds = np.unique(X_feat_column)

            for threshold in split_thresholds:
                information_gain = self.calculate_information_gain(y, X_feat_column, threshold)

                if information_gain > highest_gain_value:
                    highest_gain_value = information_gain
                    split_index = attribute_index
                    split_gain_threshold = threshold

        return split_gain_threshold, split_index

    def calculate_information_gain(self, y, X_attributes_column, split_threshold):
        """
        information gain = Entropy_{parent} - AvgEntropy_{children}

        :param y: [] array of shape (# of samples, _)
        :param X_attributes_column: [] array of shape (# of samples, _). Each array value will be
                                    the specific attribute for all samples. e.g. the RMSE for all samples
        :param split_threshold:
        :return:
        """
        y = np.array(y)
        parent_entropy = self.calculate_entropy(y)

        left_indices, right_indices = self.perform_split(X_attributes_column, split_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # calculate the weight average entropy of the children
        number_of_samples = len(y)
        num_left_samples = len(left_indices)
        num_right_samples = len(right_indices)

        entropy_left = self.calculate_entropy(y[left_indices])
        right_entropy = self.calculate_entropy(y[right_indices])

        child_entropy = ((num_left_samples/number_of_samples)*entropy_left +
                         (num_right_samples/number_of_samples)*right_entropy)

        information_gain = parent_entropy - child_entropy

        return information_gain


    def calculate_entropy(self, y_examples):
        """
         entropy = H(Y) = -SUM_k[ P(y_k)*log2(P(y_k)) ]

         with P(y_k) being the probability of random variable Y with value y_k summed over k

        :param y_examples: []
        :return: (float)
        """
        # num_of_occurrences = self.number_of_occurrences(y_examples)
        num_of_occurrences = np.bincount(y_examples)
        probabilities = num_of_occurrences / len(y_examples)
        for probability in probabilities:
            if probability > 0:
                entropy = - np.sum(probability * np.log(probability))

        return entropy

    def number_of_occurrences(self, ls):
        """
        finds number of occurrences in a list like numpy.bincount
        :param ls: []
        :return: [] with number of occurrences of each ls[n] for n in range len(ls)
        """
        unique_labels = np.unique(ls)
        occurrences = np.zeros((len(unique_labels)))
        for n in unique_labels:
            for i in range(len(ls)):
                if ls[i] == n:
                    occurrences[n] += 1
        return occurrences


    def find_most_common_label(self, list):
        """
        finds the most common value in a list
        :return:
        """
        most_occurences = np.bincount(list)
        most_common_value = 0
        if len(most_occurences) == 1:
            return 0
        else:
            for i in range(len(most_occurences)-1):
                if most_occurences[i+1] > most_occurences[i]:
                    most_common_value = i+1


        return most_common_value


    def search_tree(self, x, tree_node):
        if tree_node.is_leaf_node():
            return tree_node.node_value

        if x[tree_node.attribute] <= tree_node.threshold:
            return self.search_tree(x, tree_node.left_child)
        return self.search_tree(x, tree_node.right_child)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.search_tree(x, self.root_node))

        return np.array(predictions)

    def accuracy(self, y_predictions, y_test_data):
        correct_predictions = 0
        for i in range(len(y_predictions)):
            if y_predictions[i] == y_test_data[i]:
                correct_predictions += 1
        total_samples = len(y_predictions)
        accuracy = correct_predictions / total_samples
        return accuracy


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234
    )

    clf = DecisionTree()
    clf.__init__(maximum_tree_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('predictions: ', predictions)
    acc = clf.accuracy(predictions, y_test)
    print(acc)


# RUN TEST MAIN
# main()
