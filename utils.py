import glob
import numpy as np

def get_text_data_to_list(directory_name):
    filenames = glob.glob(f"{directory_name}/*")

    list_text = []
    for file in filenames:
        with open(file) as f:
            list_text.append(f.read())
    
    return list_text, np.array(filenames)


def get_n_most_similar_doc(similarity_matrix, n): # similarity = [Ndocs x nqueries]
    similarity_sorted_ascending =  np.argsort(similarity_matrix, axis=0)
    n_most_similarity = similarity_sorted_ascending[-n:, :] # [ndocs x nqueries]
    return n_most_similarity


def print_results(result):
    for i in range(result.shape[1]):
        print("\n***************************************************")
        print(f"10 documents the most similar to query {i + 1} are:")
        for j in range(result.shape[0]):
            print(result[j, i])    

