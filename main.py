from sklearn.feature_extraction.text import CountVectorizer
import utils

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
print(f"Dot product similarity of document and queries size: {dot_product_similarity.shape}")


# ==========================================================================================================
# List 10 most similar docs
filenames_matrix = utils.get_n_most_similar_doc(dot_product_similarity, 10) # [10, nqueries]
docs_filenames_result = docs_filenames[filenames_matrix]

utils.print_results(docs_filenames_result)





