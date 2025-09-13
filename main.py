from sklearn.feature_extraction.text import CountVectorizer
import utils
import numpy as np

# Read documents and queries
docs, docs_filenames = utils.get_text_data_to_list("docs")
queries, _ = utils.get_text_data_to_list("queries")

# ==========================================================================================================
# Extract the vocabulary out of the document collection and create a vector representation for each document
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
X = X.toarray() # (500, 2860)) = [ndocuments x vocab_size]

print(f"Vocabulary size: {X.shape[1]}")
print(f"Number of documents: {X.shape[0]}")
print(f"Document-vocab shape: {X.shape}")


# ==========================================================================================================
# Create a vector representation for each query
queries_matrix = vectorizer.transform(queries).toarray() # [nqueries x vocab_size]
print(f"Number of queries: {queries_matrix.shape}")
print(f"Query-vocab shape: {queries_matrix.shape}")


# ==========================================================================================================
# Dot product similarity = X @ queries_matrix.T (matrix of row vectors x matrix of column vectors)
dot_product_similarity = X @ queries_matrix.T # [ndocs x vocab_size] @ [vocab_size x nqueries] = [ndocs x nqueries]

# ==========================================================================================================
# Cosine product similarity = (X @ queries_matrix.T) / (norm doc * norm queries) (matrix of row vectors x matrix of column vectors)
l2_norm_docs = np.linalg.norm(X, axis=1)
l2_norm_queries = np.linalg.norm(queries_matrix, axis=1)
normalize_term = l2_norm_docs.reshape(-1, 1) * l2_norm_queries.reshape(1, -1)
cosine_similarity = dot_product_similarity / normalize_term


# Save result to file
utils.print_results(*utils.get_n_most_similar_doc(dot_product_similarity, docs_filenames, 10),
                    *utils.get_n_most_similar_doc(cosine_similarity, docs_filenames, 10),
                    )








