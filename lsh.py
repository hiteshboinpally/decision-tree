import pandas as pd
import random
import matplotlib.pyplot as plt
import time


class LSH:
    def __init__(self, data, shingle_length, permutations, num_rows_per_band, num_buckets, calc_jaccard=False):
        self.shingle_length = shingle_length
        self.num_permutations = permutations
        self.num_rows_per_band = num_rows_per_band
        self.num_buckets = num_buckets
        self.num_documents = len(data)
        #perform shingling
        #minhashing

        start_time = time.time()
        shingles_document = self.shingling(data)

        if calc_jaccard:
            self.jaccard_similarity = self.jaccard(shingles_document)
            self.jaccard_time = time.time()

        sig_matrix = self.min_hash(shingles_document)
        similar_documents = self.lsh(sig_matrix)
        self.sim_docs_set = set()
        self.get_set_of_sim_docs(similar_documents)

        self.time_taken = time.time() - start_time


    def get_runtime(self):
        return self.time_taken


    def get_set_of_sim_docs(self, similar_documents):
        for set_docs in similar_documents:
            if len(set_docs) > 1:
                self.sim_docs_set.add(frozenset(set_docs))


    def shingling(self, data):
        shingles = set()
        data_as_shingles = []
        for data_file in data:
            for i in range(len(data_file) - self.shingle_length):
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
        intersection_count = 0
        union_count = len(shingles_document)
        for i in range(len(shingles_document)):
            if shingles_document[i][0] == 1 and shingles_document[i][1] == 1:
                intersection_count += 1
        return float(intersection_count) / float(union_count)


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
                    k += 1
            signature_matrix.append(signature_matrix_row)
        return signature_matrix


    def lsh(self, sig_matrix):
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


    def is_similar(self, file1, file2):
        for sim_docs in self.sim_docs_set:
            # print('sim_docs', sim_docs)
            # print('file1', file1)
            # print('file2', file2)
            if file1 in sim_docs and file2 in sim_docs:
                return True
        return False


def permutations_vs_jaccard(data, shingle_length, rows_per_band, buckets, max_perms=15, num_trials=10):
    file_one_idx = random.randint(0,len(data))
    file_two_idx = file_one_idx
    while file_two_idx == file_one_idx:
        file_two_idx = random.randint(0,len(data))
    file_one = data[file_one_idx]
    file_two = data[file_two_idx]
    test_files = [file_one, file_two]
    jaccard_lsh = LSH(test_files, shingle_length, 1, rows_per_band, buckets, True)
    jaccard_similarity = jaccard_lsh.jaccard_similarity
    min_hash_similarities = []
    for i in range(0, max_perms, rows_per_band):
        print("calculating permuation", i)
        similarity_ct = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, i, rows_per_band, buckets)
            if curr_lsh.is_similar(file_one_idx, file_two_idx):
                similarity_ct += 1
        min_hash_similarities.append(similarity_ct / num_trials)

    print('jaccard similarity', jaccard_similarity)
    print('min hash similarities', min_hash_similarities)

    permutations = list(range(0, max_perms, rows_per_band))
    jaccard_sims = [jaccard_similarity] * len(permutations)

    plt.plot(permutations, min_hash_similarities, "ro", label='min hash')
    plt.plot(permutations, jaccard_sims, "b--", label='jaccard')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Permutations')
    plt.ylabel('Percentage of Similarity')
    plt.title('Permutations vs Similarity Percentage')
    plt.savefig('ra-plots/perms_vs_jaccard_trial_2.png')


def rows_vs_jaccard(data, shingle_length, max_rows_per_band, buckets, num_trials=10):
    file_one_idx = random.randint(0,len(data))
    file_two_idx = file_one_idx
    while file_two_idx == file_one_idx:
        file_two_idx = random.randint(0,len(data))
    # print("data", len(data))
    # print('file_one_idx', file_one_idx)
    file_one = data[file_one_idx]
    file_two = data[file_two_idx]
    test_files = [file_one, file_two]
    jaccard_lsh = LSH(test_files, shingle_length, 1, 1, 1, True)
    jaccard_similarity = jaccard_lsh.jaccard_similarity
    min_hash_similarities = []
    for i in range(1, max_rows_per_band):
        print("calculating rows per band", i)
        similarity_ct = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, i * 5, i, buckets)
            if curr_lsh.is_similar(file_one_idx, file_two_idx):
                similarity_ct += 1
        min_hash_similarities.append(similarity_ct / num_trials)

    print('jaccard similarity', jaccard_similarity)
    print('min hash similarities', min_hash_similarities)

    permutations = list(range(1, max_rows_per_band))
    jaccard_sims = [jaccard_similarity] * len(permutations)

    plt.plot(permutations, min_hash_similarities, "ro", label='min hash')
    plt.plot(permutations, jaccard_sims, "b--", label='jaccard')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Rows per Band')
    plt.ylabel('Percentage of Similarity')
    plt.title('Rows per Band vs Similarity Percentage')
    plt.savefig('ra-plots/rows_vs_jaccard.png')


def document_ct_vs_runtime(data, shingle_length, permutations, rows_per_band, buckets, num_trials=10):
    lsh_times = []
    for i in range(1, len(data)):
        print('document numbers', i)
        lsh_time = 0
        for j in range(num_trials):
            curr_lsh = LSH(data, shingle_length, permutations, rows_per_band, buckets)
            curr_lsh_time = curr_lsh.get_runtime()
            lsh_time += curr_lsh_time
        lsh_times.append(lsh_time / num_trials)

    # print('jaccard times', jaccard_times)
    # print('min hash times', lsh_times)

    document_cts = range(1, len(data))

    # plt.plot(document_cts, jaccard_times, "-r", label='Jaccard Runtime')
    plt.plot(document_cts, lsh_times, "-b", label='LSH Runtime')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Documents')
    plt.ylabel('Runtime (s)')
    plt.title('Number of Documents vs Runtime in Seconds')
    plt.savefig('ra-plots/doc_cts_vs_runtime.png')

def main():
    # data = []
    # data.append(open("hello_one.txt","r").read())
    # data.append(open("hello_two.txt","r").read())
    # data.append(open("hello_three.txt","r").read())
    # LSH(data, 3, 4, 2, 15)

    data = []
    for i in range(1, 23):
        bad_chars = ['\n', 'W']
        file_string = open("ra-data/strain" + str(i) + ".txt","r").read()
        # file_string = filter(lambda i: i not in bad_chars, file_string)
        file_string = file_string.replace('\n', '')
        file_string = file_string.replace('W', '')
        data.append(file_string)
    # file_one = random.randint(0,len(data))
    # file_two = file_one
    # while file_two == file_one:
    #     file_two = random.randint(0,len(data))
    # #data_analysis = [data[file_one], data[file_two]]
    # data_analysis = [data[0], data[3]]
    # LSH(data_analysis, 5, 100, 10, 50)
    # permutations_vs_jaccard(data, 5, 10, 50, 100, 100)
    rows_vs_jaccard(data, 5, 20, 50, 100)
    # document_ct_vs_runtime(data, 5, 50, 5, 50, 100)
    #LSH(data, 5, 100, 10, 50)


if __name__ == "__main__":
    main()