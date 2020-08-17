import pandas as pd
import random


class LSH:
    def __init__(self, data, shingle_length, permutations, num_rows_per_band, num_buckets):
        self.shingle_length = shingle_length
        self.num_permutations = permutations
        self.num_rows_per_band = num_rows_per_band
        self.num_buckets = num_buckets
        self.num_documents = len(data)
        #perform shingling
        #minhashing
        shingles_document = self.shingling(data)
        sig_matrix = self.min_hash(shingles_document)
        similar_documents = self.lsh(sig_matrix)
        self.sim_docs_set = set()
        self.get_set_of_sim_docs(similar_documents)

        print(self.sim_docs_set)


    def get_set_of_sim_docs(self, similar_documents):
        for set_docs in similar_documents:
            if len(set_docs) > 1:
                self.sim_docs_set.add(frozenset(set_docs))


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
        return signature_matrix


    def lsh(self, sig_matrix):
        buckets = {}
        similar_documents = []
        for i in range(self.num_buckets):
            similar_documents.append(set())
        for i in range(0,self.num_permutations,self.num_rows_per_band):
            current_rows = sig_matrix[i:i+self.num_rows_per_band]
            for j in range(self.num_documents):
                values_in_band = tuple([row[j] for row in current_rows])
                bucket_in = hash(values_in_band) % self.num_buckets
                if bucket_in not in buckets:
                    buckets[bucket_in] = set()
                buckets[bucket_in].add(j)
            for index,docs in buckets.items():
                if len(docs) > 1:
                    docs = frozenset(docs)
                    similar_documents[index].update(docs)
        return similar_documents


def main():
    data_one = open(r"hello_one.txt", "r")
    data_two = open(r"hello_two.txt", "r")
    data_three = open(r"hello_three.txt", "r")
    data = [data_one.read(), data_two.read(), data_three.read()]
    LSH(data, 3, 6, 2, 10)


if __name__ == "__main__":
    main()