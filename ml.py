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
        print(entropy)

        best_attribute = self._attributes[0]
        best_information_gain = 0
        for att in attributes:
            att_values = self._train_df.groupby([att])[self._classes].count()
            att_gain = entropy   
            for att_val in att_values.iteritems():
                val_count = att_val[1]
                mask_val = self._train_df[att] == att_val[0]
                data_new = self._train_df[mask_val]
                decision_values = data_new.groupby(self._classes)[att].count()
                val_prob = float(val_count) / float(total)
                decision_total = decision_values.sum()
                dec_entropy_per_attribute = 0
                for decision in decision_values.iteritems():
                    decision_count = decision[1]
                    prob_per_decision = float(decision_count)/float(decision_total)
                    dec_entropy_per_attribute -= (prob_per_decision * np.log2(prob_per_decision))
                att_gain -= (dec_entropy_per_attribute * val_prob)
            if(att_gain > best_information_gain):
                best_information_gain = att_gain
                best_attribute = att
        
            


                    
                    
                    
                
                

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