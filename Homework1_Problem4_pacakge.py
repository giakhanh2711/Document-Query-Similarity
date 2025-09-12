from pathlib import Path
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


# Read all documents & queries 
def read_files(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    names, texts = [], []
    for p in files:
        names.append(p.name)
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return names, texts

DOC_DIR = Path("/Users/yating-li/Library/CloudStorage/OneDrive-TexasA&MUniversity/PhD/003_Courses/CSCE633/Homework/homework-1/docs")
QRY_DIR = Path("/Users/yating-li/Library/CloudStorage/OneDrive-TexasA&MUniversity/PhD/003_Courses/CSCE633/Homework/homework-1/queries")
OUT_DIR = Path("/Users/yating-li/Library/CloudStorage/OneDrive-TexasA&MUniversity/PhD/003_Courses/CSCE633/Homework/homework-1/results2")
OUT_DIR.mkdir(exist_ok=True)

doc_names, doc_texts = read_files(DOC_DIR)
qry_names, qry_texts = read_files(QRY_DIR)

print(f"[Info] {len(doc_names)} documents, {len(qry_names)} queries")

# Build vocabulary on documents 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(doc_texts)   # (D × V) document-term matrix

# Transform queries into same space 
Q = vectorizer.transform(qry_texts)       # (Q × V) query-term matrix

# Similarities 
# Dot product = linear kernel (no normalization)
dot_scores = linear_kernel(Q, X)          # (Q × D)
# Cosine similarity
cos_scores = cosine_similarity(Q, X)      # (Q × D)

# Save Top-K results 
TOP_K = 10

for qi, qname in enumerate(qry_names):
    # Sort by dot product
    dot_ranking = sorted(
        zip(doc_names, dot_scores[qi]), key=lambda x: -x[1]
    )[:TOP_K]
    # Sort by cosine similarity
    cos_ranking = sorted(
        zip(doc_names, cos_scores[qi]), key=lambda x: -x[1]
    )[:TOP_K]

    # Save dot results
    out_dot = OUT_DIR / f"{qname}_dot.csv"
    with open(out_dot, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "DocName", "DotProductScore"])
        for rank, (dn, sc) in enumerate(dot_ranking, 1):
            writer.writerow([rank, dn, sc])

    # Save cosine results
    out_cos = OUT_DIR / f"{qname}_cos.csv"
    with open(out_cos, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "DocName", "CosineScore"])
        for rank, (dn, sc) in enumerate(cos_ranking, 1):
            writer.writerow([rank, dn, sc])

    print(f"Saved {out_dot} and {out_cos}")