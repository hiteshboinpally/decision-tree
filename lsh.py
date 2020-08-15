import pandas as pd
import random
class LSH:
    def __init__(self, data):
        self.shingle_length = 3
        self.num_permutations = 5
        self.num_documents = len(data)
        #perform shingling
        #minhashing
        shingles_document = self.shingling(data)
        self.min_hash(shingles_document)

    def shingling(self, data):
        shingles = set()
        data_as_shingles = []
        for data_file in data:
            for i in range(len(data_file)-self.shingle_length):
                shingles.add(data_file[i:i+self.shingle_length])
        shingles_document = []
        for shingle in shingles:
            document_in_current_shingle = []
            for data_file in data:
                if shingle in data_file:
                    document_in_current_shingle.append(1)
                else:
                    document_in_current_shingle.append(0)
            shingles_document.append(document_in_current_shingle)
        return shingles_document
    def min_hash(self, shingles_document):
        num_shingles = len(shingles_document)
        perm_indices = list(range(num_shingles))
        signature_matrix = []
        for i in range(self.num_permutations):
            random.shuffle(perm_indices)
            signature_matrix_row = [-1] * self.num_documents
            for j in range(len(perm_indices)):
                index = perm_indices.index(j)
                row = shingles_document[index]
                k = 0
                for document in row:
                    if signature_matrix_row[k] == -1 and document == 1:
                        signature_matrix_row[k] = index
                    k+=1
            signature_matrix.append(signature_matrix_row)
        print(signature_matrix)


def main():
    data_one = open("hello_one.txt","r")
    data_two = open("hello_two.txt","r")
    data_three = open("hello_three.txt","r")
    data = [data_one.read(), data_two.read(), data_three.read()]
    LSH(data)


if __name__ == "__main__":
    main()