#relevant libraries to import
import pandas as pd

class DecisionTree:
    def __init__(self, classes, attributes, data, train=0.70):
        self._classes = classes
        self._attributes = attributes

        mask = np.random.rand(len(data)) < train
        self._train_df = data[mask]
        self._test_df = data[~mask]
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
    print('test')


if __name__ == "__main__":
    main()