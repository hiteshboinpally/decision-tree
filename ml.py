# relevant libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CSE_312_Final_Project.Decision_Tree.node import Node
from sklearn.metrics import confusion_matrix
import seaborn as sns


class DecisionTree:
    """
    A Decision Tree class that allows for classification of data into categories based on
    features of the data.
    """

    def __init__(self, classes, attributes, data, train=0.70, max_height=10, get_best_val_score=False):
        """
        Constructs the actual decision tree model based on the given parameters.

        :param classes: The name of the column in the given data corresponding to classes.
        :param attributes: A list of the names of the column in the given data corresponding to
                           attributes.
        :param data: The data to create the tree as a pandas DataFrame.
        :param train: The percentage of the data to split into the training set. The remaining data
                      is split half/half into validation and testing data. Assumed to be a float.
        :param max_height: The max height of the tree to begin. Assumed to be an integer.
        """

        self._classes = classes
        self._attributes = attributes
        self._max_height = max_height

        is_train = np.random.rand(len(data)) < train
        self._train_df = data[is_train]
        leftover_df = data[~is_train]
        is_val = np.random.rand(len(leftover_df)) < (1 - train)
        self._val_df = leftover_df[is_val]
        self._test_df = leftover_df[~is_val]

        self._test_labels = list(self._test_df[self._classes].unique())

        self._curr_tree_head = Node()

        self._heights = list(range(1, self._max_height + 1))
        self._val_accs = []

        self._true_preds = []
        self._model_preds = []

        if get_best_val_score:
            self.create_best_tree()
        else:
            self.create_tree(max_height)

    def create_best_tree(self):
        """
        Determines the Decision Tree with the best accuracy out of trees with different height
        hyper-parameters.
        """
        best_val_acc = 0
        for i in self._heights:
            self.create_tree(i)
            curr_val_acc = self.get_val_accuracy()
            best_tree_head = Node()
            if type(curr_val_acc) == float:
                self._val_accs.append(curr_val_acc)
                if curr_val_acc > best_val_acc:
                    best_tree_head = self._curr_tree_head
        self._curr_tree_head = best_tree_head

    def calc_best_att(self, df):
        """
        Determines the current best attribute of the data frame, by calculating the
        information gain on each attribute that we are classifying on.

        :param df: The pandas DataFrame containing the data for which to calculate the current
                   attribute that gives the best information gain. Assumed to contain a column
                   with name = self._classes.
        :returns best_attribute: the attribute with the most information gain
        """
        classes_counts = df.groupby(self._classes)[self._attributes[0]].count()
        total = len(df)
        dec_entropy = 0
        for row in classes_counts.iteritems():
            prob = float(row[1]) / float(total)
            prob = prob * np.log2(prob)
            dec_entropy -= prob

        best_attribute = self._attributes[0]
        best_information_gain = 0
        for att in self._attributes:
            att_values = df.groupby([att])[self._classes].count()
            att_gain = dec_entropy

            for att_val in att_values.iteritems():
                val_count = att_val[1]
                mask_val = df[att] == att_val[0]
                data_new = df[mask_val]
                decision_values = data_new.groupby(self._classes)[att].count()
                val_prob = float(val_count) / float(total)
                decision_total = decision_values.sum()
                dec_entropy_per_attribute = 0
                for decision in decision_values.iteritems():
                    decision_count = decision[1]
                    prob_per_decision = float(decision_count) / float(decision_total)
                    dec_entropy_per_attribute -= (prob_per_decision * np.log2(prob_per_decision))
                att_gain -= (dec_entropy_per_attribute * val_prob)

            if att_gain > best_information_gain:
                best_information_gain = att_gain
                best_attribute = att
        return best_attribute

    def create_tree(self, height):
        """
        creates the actual Decision Tree based on entropy and information gain.
        :param height: the maximum height of the given tree. Assumed to be an int.
        """
        best_att = self.calc_best_att(self._train_df)
        self._curr_tree_head.data = best_att
        self.add_node(self._train_df, best_att, self._curr_tree_head, 1, height)

    def add_node(self, df, best_att, curr_node, curr_height, max_height):
        """
        Recursively adds nodes to the decision Tree until it reaches the max-height or classifies
        the entire training dataset.

        :param df: the pandas DataFrame that we are classifying.
        :param best_att: the current node's best attribute to split on. Assumed to be a String.
        :param curr_node: the current node of the tree that we're on. Assumed to be a Node.
        :param curr_height: the current height of the overall tree. Assumed to be an int.
        :param max_height: the maximum height of the given tree. Assumed to be an int.
        """
        unique_decs = list(df[self._classes].unique())
        curr_node.nexts = []
        curr_node.splits = []
        if len(unique_decs) == 1:
            curr_node.nexts.append(unique_decs[0])
        else:
            unique_vals = list(df[best_att].unique())
            unique_dfs = []
            for val in unique_vals:
                unique_dfs.append(df[df[best_att] == val])

            for i in range(len(unique_vals)):
                unique_df = unique_dfs[i]
                val = unique_vals[i]
                new_unique_vals = list(unique_df[self._classes].unique())

                curr_node.splits.append(val)
                if len(new_unique_vals) == 1:
                    curr_node.nexts.append(new_unique_vals[0])
                elif curr_height < max_height - 1:
                    new_best_att = self.calc_best_att(unique_df)
                    new_node = Node(new_best_att)
                    curr_node.nexts.append(new_node)
                    self.add_node(unique_df, new_best_att, new_node, curr_height + 1, max_height)
                else:
                    max_decs = unique_df.groupby(self._classes)[self._classes].count().idxmax()
                    curr_node.nexts.append(max_decs)

    def check_set(self, data, isTest=False):
        """
        Checks the accuracy of the Decision Tree based on the given data.

        :param data: The DataFrame for which to check the accuracy of the model.
        :param isTest: whether or not we're checking the test set
        :return a decimal: that represents the accuracy of the model
        """
        if len(data) == 0:
            return "No data to find the accuracy of!"
        else:
            total_correct = 0
            for index, row in data.iterrows():
                total_correct += self.start_tree(row, isTest)
            return total_correct / len(data)

    def start_tree(self, row, isTest = False):
        """
        Starts the process of passing through the tree and checking the given row against the output
        of the tree.

        :param row: The row for the tree to classify.
        :param isTest: whether or not we're checking the test set
        :return method call: calls recursive method that is used to pass through the nodes of the decision tree, while
                             comparing the given value to the output.
        """
        return self.pass_thru_tree(row, self._curr_tree_head, isTest)

    def pass_thru_tree(self, row, node, isTest):
        """
        Recursively passes through the tree node by node and attempts to classify the given row in
        the instance's Decision Tree.

        :param row: The row for the tree to classify.
        :param node: the current node of the tree.
        :param isTest: whether or not we're checking the test set
        :return recursive method call:
                in which the node parameter is updated to represents the next child node in the tree
        """
        if type(node) == str:
            row_desc = row[self._classes]
            if isTest:
                self._model_preds.append(node)
                self._true_preds.append(row_desc)
            if node == row_desc:
                return 1
            else:
                return 0
        else:
            curr_att = node.data
            val_in_row = row[curr_att]
            if val_in_row not in node.splits:
                next_node_index = np.random.randint(len(node.nexts))
            else:
                next_node_index = node.splits.index(val_in_row)
            return self.pass_thru_tree(row, node.nexts[next_node_index], isTest)

    def get_val_accuracy(self):
        """
        Returns the accuracy of the model on the validation set.

        :return a decimal: that represents the accuracy of the validation set.
        """
        return self.check_set(self._val_df)

    def get_train_accuracy(self):
        """
        Returns the accuracy of the model on the training set.

        :return a decimal: that represents the accuracy of the train set.
        """
        return self.check_set(self._train_df)

    def get_test_accuracy(self):
        """
        Returns the accuracy of the model on the testing set.

        :return a decimal: that represents the accuracy of the test set.
        """
        return self.check_set(self._test_df, True)

    def get_heights_and_val_accs(self):
        """
        Returns the heights of the decision tree and the corresponding validation accuracies
        of the model

        :return the heights of the decision tree and the corresponding validation accuracies
        of the model
        """
        return self._heights, self._val_accs

    def get_confusion_matrix(self):
        """
        Creates the confusion matrix which displays True positives, True Negatives, False Positives
        and False Negatives.
        :return: a matrix: thats rows are the number of data points classified by the model to that attribute
                            and the columns are the number of data points of a given attribute.
        """
        if len(self._true_preds) == 0:
            self.get_test_accuracy()
        return confusion_matrix(self._true_preds, self._model_preds, labels=self._test_labels)

    def inorder(self, node):
        """
        Inorder traversal of the n-ary Decision Tree where we print out the value at each node
        :param node: the curr node of the Decision Tree to print
        """
        if node == None:
            return
        elif type(node) == str:
            print(node)
        else:
            total = len(node.nexts)
            for i in range(total - 1):
                self.inorder(node.nexts[i])

            print(node.data, node.splits)

            self.inorder(node.nexts[total - 1])


def set_months(date):
    """
    Converts a date (Year-Month-Day) to a season for the Decision Tree

    :param date: the date the data point occurred at
    :return a category: represents what season that data point occurred in.
    """
    month_num = int(date[5:7])
    if month_num < 4:
        return "Winter"
    elif month_num < 7:
        return "Spring"
    elif month_num < 10:
        return "Summer"
    else:
        return "Winter"


def set_faren(temp, suffix):
    """
    Converts a temperature (in Fahrenheit) to a category for the Decision Tree

    :param temp: the temperature on a given day
    :param suffix: the name of the category that we add to Low, Mid, and High (i.e. Temperature)
    :return a category: represents whether the temp is low (< 33), medium (<70), or high.
    """
    temp = int(temp)
    if temp < 33:
        return "Low " + suffix
    elif temp < 70:
        return "Mid " + suffix
    else:
        return "High " + suffix


def set_temps(temp):
    """
    Converts the temperature (in Fahrenheit) on a given day to a category for the Decision Tree

    :param temp: the temperature on a given day
    :return a category: that returns Low temperature, Mid temperature, or High temperature
    """
    return set_faren(temp, "temperature")


def set_dew_points(temp):
    """
    Converts a dew point (in Fahrenheit) to a category for the Decision Tree

    :param temp: the temperature on a given day
    :return a category: that returns Low dew points, Mid dew points, or High dew points
                            or Unknown dew points if there was no value in that data point
    """
    if temp == '-':
        return "Unknown dew points"
    else:
        return set_faren(temp, "dew points")


def set_humid_percent(percent):
    """
    Converts a the percentage humidity to a category for the Decision Tree

    :param percent: the percent humidity on a given day
    :return a category: that returns Low humidity, Mid humidity, or High humidity
                            or Unknown humidity if there was no value in that data point
    """
    if percent == '-':
        return "Unknown humidity"
    percent = int(percent)
    if percent < 33:
        return "Low humidity"
    elif percent < 66:
        return "Mid humidity"
    else:
        return "High humidity"


def set_sea_level(pressure):
    """
    Converts a temperature (in Fahrenheit) to a category for the Decision Tree

    :param pressure: the percent pressure on a given day
    :return a category: that returns Low Sea Pressure, Mid Sea Pressure, or High Sea Pressure
                            or Unknown Sea Pressure if there was no value in that data point
    """
    if pressure == '-':
        return "Unknown Sea Pressure"

    pressure = float(pressure)
    if pressure < 29.883:
        return "Low Sea Pressure"
    elif pressure < 30.3563:
        return "Mid Sea Pressure"
    else:
        return "High Sea Pressure"


def set_visibility(visibility):
    """
    Converts visibilty  (in miles) to a category for the Decision Tree

    :param visibility: the number of miles of visibility pressure on a given day
    :return a category: that returns Low Visibility, Mid Visibility, or High Visibility
                            or Unknown Visibility if there was no value in that data point
    """
    if visibility == '-':
        return "Unknown Visibility"

    visibility = float(visibility)
    if visibility < 3.33:
        return "Low Visibility"
    elif visibility < 6.66:
        return "Mid Visibility"
    else:
        return "High Visibility"


def set_wind_mph(wind_mph):
    """
    Converts the average wind speed (in MPH) to a category for the Decision Tree

    :param wind_mph: the wind speed on a given day
    :return a category: that returns Low wind, Mid wind, or High wind
                            or Unknown wind if there was no value in that data point
    """
    if wind_mph == '-':
        return "Unknown Wind"
    wind_mph = float(wind_mph)
    if wind_mph < 4:
        return "Low Wind"
    elif wind_mph < 7:
        return "Mid Wind"
    else:
        return "High Wind"


def setup_weather_data():
    """
    Calls all the methods that replace the data within each point within a category and returns a new
    list with all data points described in categories.

    :return a list: of all the attributes of each weather data point
    """
    data = pd.read_csv("ml-data/austin_weather_data.csv")
    updated_df = [data["Date"].apply(set_months), data["TempHighF"].apply(set_temps), data["TempAvgF"].apply(set_temps),
                  data["TempLowF"].apply(set_temps), data["DewPointHighF"].apply(set_dew_points),
                  data["DewPointAvgF"].apply(set_dew_points), data["DewPointLowF"].apply(set_dew_points),
                  data["HumidityHighPercent"].apply(set_humid_percent),
                  data["HumidityAvgPercent"].apply(set_humid_percent),
                  data["HumidityLowPercent"].apply(set_humid_percent),
                  data["SeaLevelPressureHighInches"].apply(set_sea_level),
                  data["SeaLevelPressureAvgInches"].apply(set_sea_level),
                  data["SeaLevelPressureLowInches"].apply(set_sea_level),
                  data["VisibilityHighMiles"].apply(set_visibility), data["VisibilityAvgMiles"].apply(set_visibility),
                  data["VisibilityLowMiles"].apply(set_visibility), data["WindHighMPH"].apply(set_wind_mph),
                  data["WindAvgMPH"].apply(set_wind_mph)]

    final_data = pd.DataFrame(updated_df).T

    return final_data


def plot_heights_vs_val_accs(classes, features, data, train_perc=0.7, max_height=10, num_trials=10):
    """
    Creates graph with the Decision Tree Height as the x-axis and validation accuracy as the y-axis

    :param classes: The name of the column in the given data corresponding to classes.
    :param features: A list of the names of the column in the given data corresponding to
                           attributes.
    :param data: the set of all data points that contains info on Austin weather
    :param train_perc: percentage of total data set that is dedicated to the training set
    :param max_height: Maximum height of the Decision Tree (Hyper-paramter)
    :param num_trials: Number of times we will run the Decision Tree with given max_height parameter
    """

    heights = range(max_height)
    all_val_accs = []
    print("plotting heights of tree versus validation accuracy...")
    for i in range(num_trials):
        print("trial #" + str(i))
        tree = DecisionTree(classes, features, data, train_perc, max_height, True)
        x, val_accs = tree.get_heights_and_val_accs()
        all_val_accs.append(val_accs)

    results = pd.DataFrame(all_val_accs, columns=heights)
    results = results.mean()
    plt.plot(list(results.index), results.values, "-b")
    plt.plot(list(results.index), results.values, "ro")
    plt.xlabel('Height of Tree')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Height of Tree vs Average Validation Accuracy')
    plt.savefig('ml-plots/height_vs_vals.png')


def plot_train_perc_vs_test_accs(classes, features, data, max_height=10, num_trials=10):
    """
    Creates graph with the Training Percentage as the x-axis and the Test Accuracy as the y-axis

    :param classes: The name of the column in the given data corresponding to classes.
    :param features: A list of the names of the column in the given data corresponding to
                           attributes.
    :param data: the set of all data points that contains info on Austin weather
    :param max_height: Maximum height of the Decision Tree (Hyper-paramter)
    :param num_trials: Number of times we will run the Decision Tree with given max_height parameter
    """
    train_perc = np.linspace(0.1, 1, 11)
    all_test_accs = []
    print("plotting training percentage versus testing accuracy...")
    for i in range(num_trials):
        print("trial #" + str(i))
        curr_test_accs = []
        for j in train_perc:
            tree = DecisionTree(classes, features, data, j, max_height)
            curr_test_accs.append(tree.get_test_accuracy())
        all_test_accs.append(curr_test_accs)

    results = pd.DataFrame(all_test_accs, columns=list(train_perc))
    results = results.mean()

    plt.plot(list(results.index), results.values, "-b")
    plt.plot(list(results.index), results.values, "ro")
    plt.xlabel('Percentage of Data for Training Set')
    plt.ylabel('Average Test Accuracy')
    plt.title('Percentage of Training Set Data vs Test Accuracy')
    plt.savefig('ml-plots/train_perc_vs_tests.png')


def plot_confusion_matrix(tree):
    """
    Creates a heat plot for the confusion matrix

    :param tree: The Decision Tree
    :return: heat plot of the confusion matrix
    """
    print("plotting confusion matrix...")
    cf_matrix = tree.get_confusion_matrix()
    categories = tree._test_labels

    cf_matrix_df = pd.DataFrame(cf_matrix, columns=categories, index=categories)
    cf_matrix_df.index.name = 'Actual'
    cf_matrix_df.columns.name = 'Predicted'

    sns.set(font_scale=1.4)
    plt.figure(figsize=(17, 7))
    ax = sns.heatmap(cf_matrix_df, cmap='Blues', annot=True, fmt='g')
    plt.yticks(rotation=0)
    ax.invert_yaxis()
    plt.title('Confusion Matrix')
    plt.savefig('ml-plots/confusion_matrix.png')


"""
def sample_tree_algo():
    sample_data = pd.read_csv('ml-data/sample.txt')
    sample_cols = list(sample_data.columns)
    sample_tree = DecisionTree(sample_cols[4], sample_cols[0:4], sample_data, 1, max_height=2)
    sample_tree.inorder(sample_tree._curr_tree_head)
"""


def main():
    # sample_tree_algo()

    weather_data = setup_weather_data()
    weather_cols = list(weather_data.columns)
    weather_classes = weather_cols[2]
    weather_features = weather_cols[0:2]
    weather_features.extend(weather_cols[3:])
    weather_tree = DecisionTree(weather_classes, weather_features, weather_data, 0.7, 7)
    print("accuracy", weather_tree.get_test_accuracy())
    # weather_tree.inorder(weather_tree._curr_tree_head)
    # plot_confusion_matrix(weather_tree)
    # plot_heights_vs_val_accs(weather_classes, weather_features, weather_data, max_height=15, num_trials=15)
    # plot_train_perc_vs_test_accs(weather_classes, weather_features, weather_data, max_height=7, num_trials=10)

if __name__ == "__main__":
    main()
