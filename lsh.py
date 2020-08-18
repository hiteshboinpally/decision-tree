import pandas as pd
import random


class LSH:
    def __init__(self, data, shingle_length, permutations, num_rows_per_band, num_buckets):
        """
        An Locality Sensitive Hashing program that places similar documents in the same hash bucket with high
        probability and dissimilar documents in different buckets with high probability.

        :param data: a list of documents or genomes
        :param shingle_length: the length of each gram/shingle of the document/data point
        :param permutations: the number of permutations that will be used in the min-Hash Function
        :param num_rows_per_band: the number of rows in the band where we will hash column values into
        :param num_buckets: the number of buckets in each band
        """
        self.shingle_length = shingle_length
        self.num_permutations = permutations
        self.num_rows_per_band = num_rows_per_band
        self.num_buckets = num_buckets
        self.num_documents = len(data)
        #perform shingling
        #minhashing
        shingles_document = self.shingling(data)
        jaccard_similarity = self.jaccard(shingles_document)
        print(jaccard_similarity)
        sig_matrix = self.min_hash(shingles_document)
        similar_documents = self.lsh(sig_matrix)
        self.sim_docs_set = set()


    def get_set_of_sim_docs(self, similar_documents):
        """
        Finds the similar documents within a the original list of documents

        :param similar_documents: list of similar documents or genomes
        """
        for set_docs in similar_documents:
            if len(set_docs) > 1:
                self.sim_docs_set.add(frozenset(set_docs))


    def shingling(self, data):
        """
        Creates k length shingles for all the documents in the original data file

        :param data: list of documents or genomes
        :return: a list of all the shingles in a given document
        """
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

    def jaccard(self, shingles_document):
        """
        Calculates the Jaccard Index in a given document, or the intersection of all shingles divided by the
        union of all shingles in a document or genome.

        :param shingles_document: a matrix where the rows are the shingles and the columns are the different files
        :return: a float: that represents the Jaccard Index of a given document or genome
        """
        intersection_count = 0
        union_count = len(shingles_document)
        for i in range(len(shingles_document)):
            if shingles_document[i][0] == 1 and shingles_document[i][1] == 1:
                intersection_count += 1
        return float(intersection_count) / float(union_count)

    def min_hash(self, shingles_document):
        """
        Calculates and returns the min hash signature matrix

        :param shingles_document: a matrix where the rows are the shingles and the columns are the different files
        :return signature matrix: a matrix where the rows represent different files and if the columns are similar
                                    there is a high probability that the documents are similar too
        """
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
                    k += 1
            signature_matrix.append(signature_matrix_row)
        return signature_matrix


    def lsh(self, sig_matrix):
        """
        Returns a list of sets of similar documents or genomes from the given data set

        :param sig_matrix: a matrix where the rows represent different files and if the columns are similar
                                    there is a high probability that the documents are similar too
        :return: a list: that contains sets of similar documents
        """
        buckets = {}
        similar_documents = []
        for i in range(self.num_buckets):
            similar_documents.append(set())
        for i in range(0,self.num_permutations,self.num_rows_per_band):
            buckets.clear()
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
    # data = []
    # data.append(open("hello_one.txt","r").read())
    # data.append(open("hello_two.txt","r").read())
    # data.append(open("hello_three.txt","r").read())
    # LSH(data, 3, 4, 2, 15)

    data = []
    for i in range(1,23):
        bad_chars = ['\n', 'W']
        file_string = open("ra-data/strain"+str(i)+".txt","r").read()
        file_string = filter(lambda i: i not in bad_chars, file_string)
        data.append(file_string)
    file_one = random.randint(0,len(data))
    file_two = file_one
    while file_two == file_one:
        file_two = random.randint(0,len(data))
    #data_analysis = [data[file_one], data[file_two]]
    data_analysis = [data[0], data[3]]
    LSH(data_analysis, 5, 100, 10, 50)
    #LSH(data, 5, 100, 10, 50)

        
    


if __name__ == "__main__":
    main()