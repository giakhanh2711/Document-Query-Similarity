import glob
import numpy as np
import pandas as pd
import os

def get_text_data_to_list(directory_name):
    filenames = glob.glob(f"{directory_name}/*")

    list_text = []
    for file in filenames:
        with open(file) as f:
            list_text.append(f.read())
    
    return list_text, np.array(filenames)


def get_n_most_similar_doc(similarity_matrix, filenames, n): # similarity = [Ndocs x nqueries]
    similarity_sorted_ascending =  np.argsort(similarity_matrix, axis=0)
    n_most_similarity = similarity_sorted_ascending[-n:, :][::-1] # [ndocs x nqueries]
    return filenames[n_most_similarity], np.sort(similarity_matrix, axis=0)[-n:, :][::-1]


def print_results(dot_doc, dot_sim, cosine_doc, cosine_sim):
    os.makedirs("results", exist_ok=True)

    for i in range(dot_doc.shape[1]):
        data = {"Rank": list(range(1, 11)),
            "Dot Document": [x.split("\\")[1] for x in dot_doc[:, i]],
            "Dot Product Similarity": dot_sim[:, i],
            "Cosine Document": [x.split("\\")[1] for x in cosine_doc[:, i]],
            "Cosine Product Similarity": cosine_sim[:, i]
            }
           
        df = pd.DataFrame(data)
        df.to_csv(os.path.join("results", f"{i + 1}.csv"), index=False)

