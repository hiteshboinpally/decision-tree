#relevant libraries to import
import pandas as pd
from binarytree import Node 
import numpy as np

class DecisionTree:
    def __init__(self, classes, attributes, data, train=0.70):
        self._classes = classes
        self._attributes = attributes
        mask = np.random.rand(len(data)) < train
        self._train_df = data[mask]
        self._test_df = data[~mask]
        
        classes_counts = self._train_df.groupby(classes)[self._attributes[0]].count()
        total = len(data)
        entropy = 0
        for row in classes_counts.iteritems():
            prob = float(row[1])/float(total)
            prob = prob * np.log2(prob)
            entropy -= prob

        best_attribute = self._attributes[0]
        best_information_gain = 0
        for att in self._attributes:
            att_counts = pd.crosstab(self._train_df[att],self._train_df[classes])
            for row in att_counts.iterrows():
                totalNumer = 0
                for decision in row:
                    print(decision)
                
            

                

        #for att in attributes:
            

        #Go through each row -> move through tree according to data attributes of row
        # determine information gain on all three attributes and find greatest
        #Store value we split at 

        # data_length = len(data)
        # train_set_indices = set()
        # test_set_indices = set()
        # while len(train_set_indices) < (data_length * train):
        #   train_set_indices.add(np.random.randint(0, data_length))
        # for i in range(data_length):
        #   if i not in train_set_indices:
        #     test_set_indices.add(i)

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