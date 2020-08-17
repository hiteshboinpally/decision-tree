# relevant libraries to import
import pandas as pd
import numpy as np
from node import Node


class DecisionTree:
    """
    A Decision Tree class that allows for classification of data into categories based on
    features of the data.
    """
    def __init__(self, classes, attributes, data, train=0.70, max_height=10):
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
        self._tree_head = Node()

        self.create_tree()
        val_score = self.check_set(self._val_df)
        # print('val_score', val_score)
        while ((type(val_score) != str) and val_score < self._desired_acc):
            self._max_height -= 1
            self.create_tree()
            val_score = self.check_set(self._val_df)
            # print('val_score', val_score)


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
            # print(att)
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


    def create_tree(self):
        """
        creates the actual Decision Tree based on entropy and information gain.
        """
        best_att = self.calc_best_att(self._train_df)
        self._tree_head.data = best_att
        self.add_node(self._train_df, best_att, self._tree_head, 1)


    def add_node(self, df, best_att, curr_node, curr_height):
        """
        Recursively adds nodes to the decision Tree until it reaches the max-height or classifies
        the entire training dataset.

        :param df: the pandas DataFrame that we are classifying.
        :param best_att: the current node's best attribute to split on. Assumed to be a String.
        :param curr_node: the current node of the tree that we're on. Assumed to be a Node.
        :param curr_height: the current height of the overall tree. Assumed to be an int.
        """
        unique_decs = list(df[self._classes].unique())
        curr_node.nexts = []
        curr_node.splits = []
        if len(unique_decs) == 1:
            curr_node.nexts.append(unique_decs[0])
        elif curr_height <= self._max_height:
            unique_vals = list(df[best_att].unique())
            unique_dfs = []
            for val in unique_vals:
                unique_dfs.append(df[df[best_att] == val])

            # print("unique df length", len(unique_dfs), "unique val length", len(unique_vals))
            for i in range(len(unique_vals)):
                unique_df = unique_dfs[i]
                val = unique_vals[i]
                new_unique_vals = list(unique_df[self._classes].unique())

                curr_node.splits.append(val)
                if len(new_unique_vals) == 1:
                    curr_node.nexts.append(new_unique_vals[0])
                else:
                    new_best_att = self.calc_best_att(unique_df)
                    new_node = Node(new_best_att)
                    curr_node.nexts.append(new_node)
                    self.add_node(unique_df, new_best_att, new_node, curr_height + 1)
        else:
            max_decs = df.groupby(self._classes)[self._classes].count().idxmax()
            curr_node.nexts.append(max_decs)


    def print_sub_tree(self, node):
        """
        Recursively prints out the tree starting at the current node.

        :param node: The current node of the tree to print. Assumed to be a Node object.
        """
        if type(node) == str:
            print(node)
        else:
            print(node.data)
            print(node.splits)
            if type(node.nexts) != str:
                for i in range(len(node.nexts)):
                    next_node = node.nexts[i]
                    self.print_sub_tree(next_node)


    def print_tree(self):
        """
        Prints out the decision tree.
        """
        self.print_sub_tree(self._tree_head)


    def check_set(self, data):
        """
        Checks the accuracy of the Decision Tree based on the given data.

        :param data: The DataFrame for which to check the accuracy of the model.
        :returns a decimal: that represents the accuracy of the model
        """
        if len(data) == 0:
            return "No data to find the accuracy of!"
        else:
            total_correct = 0
            for index, row in data.iterrows():
                # print(row)
                total_correct += self.start_tree(row)

            # print(total_correct)
            return total_correct / len(data)


    def start_tree(self, row):
        """
        Starts the process of passing through the tree and checking the given row against the output
        of the tree.

        :param row: The row for the tree to classify.
        :returns method call:
                             calls recursive method that is used to pass through the nodes of the decision tree, while
                             comparing the given value to the output.
        """
        return self.pass_thru_tree(row, self._tree_head)


    def pass_thru_tree(self, row, node):
        """
        Recursively passes through the tree node by node and attempts to classify the given row in
        the instance's Decision Tree.

        :param row: The row for the tree to classify.
        :param node: the current node of the tree.
        :returns recursive method call:
                in which the node parameter is updated to represents the next child node in the tree
        """
        # print(type(row))
        if type(node) == str:
            # print("desc", node)
            row_desc = row[self._classes]
            # print("row desc", row_desc)
            if node == row_desc:
                # print("we're good!")
                return 1
            else:
                # print("rip")
                return 0
        else:
            # print("node", node)
            curr_att = node.data
            val_in_row = row[curr_att]
            if val_in_row not in node.splits:
                next_node_index = np.random.randint(len(node.nexts))
            else:
                next_node_index = node.splits.index(val_in_row)
            # print("val_in_row", val_in_row)
            # print("next_node_index", next_node_index)
            return self.pass_thru_tree(row, node.nexts[next_node_index])


    def get_val_accuracy(self):
        """
        Returns the accuracy of the model on the validation set.
        :returns a decimal: that represents the accuracy of the validation set.
        """
        return self.check_set(self._val_df)


    def get_train_accuracy(self):
        """
        Returns the accuracy of the model on the training set.
        :returns a decimal: that represents the accuracy of the train set.
        """
        return self.check_set(self._train_df)


    def get_test_accuracy(self):
        """
        Returns the accuracy of the model on the testing set.
        :returns a decimal: that represents the accuracy of the test set.
        """
        return self.check_set(self._test_df)




def set_months(date):
    """
    Converts a date (Year-Month-Day) to a season for the Decision Tree

    :param date: the date the data point occurred at
    :returns a category: represents what season that data point occurred in.
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
    :returns a category: represents whether the temp is low (< 33), medium (<70), or high.
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
    :returns a category: that returns Low temperature, Mid temperature, or High temperature
    """
    return set_faren(temp, "temperature")


def set_dew_points(temp):
    """
    Converts a dew point (in Fahrenheit) to a category for the Decision Tree

    :param temp: the temperature on a given day
    :returns a category: that returns Low dew points, Mid dew points, or High dew points
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
    :returns a category: that returns Low humidity, Mid humidity, or High humidity
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
    :returns a category: that returns Low Sea Pressure, Mid Sea Pressure, or High Sea Pressure
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
    :returns a category: that returns Low Visibility, Mid Visibility, or High Visibility
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
    :returns a category: that returns Low wind, Mid wind, or High wind
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

    :returns a category: that returns Low dew points, Mid dew points, or High dew points
                            or Unknown dew points if there was no value in that data point
    """
    data = pd.read_csv("ml-data/austin_weather_data.csv")
    updated_df = []

    updated_df.append(data["Date"].apply(set_months))
    updated_df.append(data["TempHighF"].apply(set_temps))
    updated_df.append(data["TempAvgF"].apply(set_temps))
    updated_df.append(data["TempLowF"].apply(set_temps))
    updated_df.append(data["DewPointHighF"].apply(set_dew_points))
    updated_df.append(data["DewPointAvgF"].apply(set_dew_points))
    updated_df.append(data["DewPointLowF"].apply(set_dew_points))
    updated_df.append(data["HumidityHighPercent"].apply(set_humid_percent))
    updated_df.append(data["HumidityAvgPercent"].apply(set_humid_percent))
    updated_df.append(data["HumidityLowPercent"].apply(set_humid_percent))
    updated_df.append(data["SeaLevelPressureHighInches"].apply(set_sea_level))
    updated_df.append(data["SeaLevelPressureAvgInches"].apply(set_sea_level))
    updated_df.append(data["SeaLevelPressureLowInches"].apply(set_sea_level))
    updated_df.append(data["VisibilityHighMiles"].apply(set_visibility))
    updated_df.append(data["VisibilityAvgMiles"].apply(set_visibility))
    updated_df.append(data["VisibilityLowMiles"].apply(set_visibility))
    updated_df.append(data["WindHighMPH"].apply(set_wind_mph))
    updated_df.append(data["WindAvgMPH"].apply(set_wind_mph))
    final_data = pd.DataFrame(updated_df).T

    return final_data

def main():
    # data = pd.read_csv("ml-data/sample.txt")
    # cols = list(data.columns)
    # tree_one = DecisionTree(cols[4], cols[0:4], data)
    # tree_one.print_tree()
    # print("train df \n", tree_one._train_df)
    # print("val df \n", tree_one._val_df)
    # print("test df \n", tree_one._test_df)
    # print("accuracy", tree_one.get_test_accuracy())

    weather_data = setup_weather_data()
    weather_cols = list(weather_data.columns)
    weather_classes = weather_cols[2]
    weather_features = weather_cols[0:2]
    weather_features.extend(weather_cols[3:])
    tree = DecisionTree(weather_classes, weather_features, weather_data, 0.7, 10)
    # tree.print_tree()
    print("accuracy", tree.get_test_accuracy())


if __name__ == "__main__":
    main()
