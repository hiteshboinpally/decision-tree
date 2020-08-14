class Node:
    def __init__(self, data=None, next=None):
        self.data = data # Stores the attribute being split on
        self.nexts = None # Stores the next nodes in the tree
        self.splits = None # Stores the values of the attributes corresponding to each node