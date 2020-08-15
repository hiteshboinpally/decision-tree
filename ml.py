# relevant libraries to import
import pandas as pd
import numpy as np
from node import Node


class DecisionTree:
    def __init__(self, classes, attributes, data, train=0.70, max_height=10, desired_acc=0.75):
        self._classes = classes
        self._attributes = attributes
        self._max_height = max_height
        self._desired_acc = desired_acc

        is_train = np.random.rand(len(data)) < train
        self._train_df = data[is_train]
        leftover_df = data[~is_train]
        is_val = np.random.rand(len(leftover_df)) < (1 - train)
        self._val_df = leftover_df[is_val]
        self._test_df = leftover_df[~is_val]
        self._tree_head = Node()

        self.create_tree()
        val_score = self.check_set(self._val_df)
        print('val_score', val_score)
        while ((type(val_score) != str) and val_score < self._desired_acc):
            self._max_height -= 1
            self.create_tree()
            val_score = self.check_set(self._val_df)
            print('val_score', val_score)

    def calc_best_att(self, df):
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

    def create_tree(self):
        best_att = self.calc_best_att(self._train_df)
        self._tree_head.data = best_att
        self.add_node(self._train_df, best_att, self._tree_head, 1)

    def add_node(self, df, best_att, curr_node, curr_height):
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
        self.print_sub_tree(self._tree_head)

    def check_set(self, data):
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
        return self.pass_thru_tree(row, self._tree_head)

    def pass_thru_tree(self, row, node):
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

    def get_test_accuracy(self):
        return self.check_set(self._train_df)


def main():
    data = pd.read_csv("sample.txt")
    cols = list(data.columns)
    tree_one = DecisionTree(cols[4], cols[0:4], data)
    tree_one.print_tree()
    print("train df \n", tree_one._train_df)
    print("val df \n", tree_one._val_df)
    print("test df \n", tree_one._test_df)
    print("accuracy", tree_one.get_test_accuracy())


if __name__ == "__main__":
    main()
