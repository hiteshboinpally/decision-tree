#relevant libraries to import
import pandas as pd
import numpy as np
from node import Node

class DecisionTree:
    def __init__(self, classes, attributes, data, train=0.70):
        self._classes = classes
        self._attributes = attributes
        mask = np.random.rand(len(data)) < train
        self._train_df = data[mask]
        self._test_df = data[~mask]
        self._tree_head = Node()

        self.create_tree()


    def calc_best_att(self, df):
        classes_counts = df.groupby(self._classes)[self._attributes[0]].count()
        total = len(df)
        entropy = 0
        for row in classes_counts.iteritems():
            prob = float(row[1])/float(total)
            prob = prob * np.log2(prob)
            entropy -= prob

        best_attribute = self._attributes[0]
        best_information_gain = 0
        for att in self._attributes:
            att_values = df.groupby([att])[self._classes].count()
            att_gain = entropy

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
                    prob_per_decision = float(decision_count)/float(decision_total)
                    dec_entropy_per_attribute -= (prob_per_decision * np.log2(prob_per_decision))
                att_gain -= (dec_entropy_per_attribute * val_prob)

            if (att_gain > best_information_gain):
                best_information_gain = att_gain
                best_attribute = att
        return best_attribute


    def create_tree(self):
        best_att = self.calc_best_att(self._train_df)
        print("first best attribute", best_att)
        self._tree_head.data = best_att
        self.add_node(self._train_df, best_att, self._tree_head)


    def add_node(self, df, best_att, curr_node):
        unique_vals = list(df[self._classes].unique())
        curr_node.next = []
        if len(unique_vals) == 1:
            curr_node.next.append(unique_vals[0])
            print("final class:", unique_vals[0])
        else:
            unique_vals = list(df[best_att].unique())
            unique_dfs = []
            for val in unique_vals:
                unique_dfs.append(df[df[best_att] == val])

            for unique_df in unique_dfs:
                unique_vals = list(unique_df[self._classes].unique())
                if len(unique_vals) == 1:
                    curr_node.next.append(unique_vals[0])
                    print("final class:", unique_vals[0])
                else:
                    new_best_att = self.calc_best_att(unique_df)
                    new_node = Node(new_best_att)
                    curr_node.next.append(new_node)
                    print("new best att:", new_best_att)
                    self.add_node(unique_df, new_best_att, new_node)


    def calc_information_gain(self, attribute):
        # calculate the information gain for the given attribute
        pass


    def calc_entropy(self, attribute):
        # calculate the entropy for the given attribute
        pass


def main():
    data = pd.read_csv("sample.txt")
    cols = list(data.columns)
    tree_one = DecisionTree(cols[4], cols[0:4], data, 1)




if __name__ == "__main__":
    main()