"""
Document similarity with the Vector Space Model (TECH 400, Week 3).

This program:
  1. Loads a collection of documents (.txt, .docx or .pdf) from a folder.
  2. Preprocesses the text (lowercase, tokenize, remove stop words, stem).
  3. Builds a TF-IDF weighted vector for every document.
  4. Normalizes each vector to unit length (cosine normalization).
  5. Computes the cosine similarity between every pair of documents.
  6. Ranks the documents against free-text queries.

The TF-IDF and cosine calculations are written by hand so that each step is
easy to follow. scikit-learn is only used at the end to cross-check the
results, and matplotlib only to draw the heatmap. Both are optional.

Usage:
    python vsm_similarity.py                      # full report with sample queries
    python vsm_similarity.py "neural network"     # also rank your own query
"""

import math
import os
import re
import sys
from collections import Counter

from nltk.stem import PorterStemmer

# Folder that holds the document collection and folder for generated files.
DOCUMENTS_FOLDER = "documents"
OUTPUT_FOLDER = "output"

# Queries used for the ranking demonstration.
SAMPLE_QUERIES = [
    "ranking documents with term weighting and cosine similarity",
    "neural networks learn from training data",
    "greenhouse gas emissions and solar power",
    "goalkeeper scores in the world cup",
]

# A standard English stop word list (Manning et al., 2008, ch. 2).
STOP_WORDS = set("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each even few for from further had has have having he her here hers
herself him himself his how however i if in into is it its itself just many
may me might more most much must my myself no nor not now of off on once only
or other others our ours ourselves out over own same she should since so some
such than that the their theirs them themselves then there these they this
those through thus to too under until up upon us used using very was we were
what when where which while who whom why will with within without would you
your yours yourself yourselves
""".split())

stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# 1. DOCUMENT LOADING
# ---------------------------------------------------------------------------

def read_file(file_path):
    """Return the plain text of a .txt, .docx or .pdf file."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as opened_file:
            return opened_file.read()

    if file_path.endswith(".docx"):
        import docx  # python-docx, only needed for Word files
        return "\n".join(p.text for p in docx.Document(file_path).paragraphs)

    if file_path.endswith(".pdf"):
        import pypdf  # only needed for PDF files
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""


def load_documents(folder_name):
    """Read every supported file in the folder and give each one a document ID.

    Returns a dictionary shaped like:
        {1: {"filename": "doc1_....txt", "title": "Information retrieval",
             "text": "..."}, ...}
    """
    documents = {}
    supported = (".txt", ".docx", ".pdf")

    # Sorting keeps the order predictable, so doc1_... always becomes ID 1.
    file_names = sorted(f for f in os.listdir(folder_name) if f.endswith(supported))

    for document_id, file_name in enumerate(file_names, start=1):
        text = read_file(os.path.join(folder_name, file_name)).strip()
        title = text.splitlines()[0] if text else file_name
        documents[document_id] = {"filename": file_name, "title": title, "text": text}

    return documents


# ---------------------------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_text(text):
    """Turn raw text into a list of index terms.

      1. lowercase               -> "Search" and "search" become the same
      2. keep alphabetic words   -> numbers and punctuation are dropped
      3. remove stop words       -> "the", "and", "of" carry no topic
      4. Porter stemming         -> "retrieval", "retrieve" -> "retriev"
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    return [stemmer.stem(token) for token in tokens
            if token not in STOP_WORDS and len(token) > 1]


# ---------------------------------------------------------------------------
# 3. TF-IDF WEIGHTING
# ---------------------------------------------------------------------------

def compute_term_frequencies(documents):
    """Count how often each term occurs in each document: {doc_id: Counter}."""
    return {doc_id: Counter(preprocess_text(doc["text"]))
            for doc_id, doc in documents.items()}


def compute_idf(term_frequencies):
    """Inverse document frequency:  idf(t) = log10(N / df(t)).

    df(t) is the number of documents that contain term t. A term found in
    every document gets idf = 0, so it has no effect on similarity.
    """
    number_of_documents = len(term_frequencies)
    document_frequency = Counter()
    for counts in term_frequencies.values():
        document_frequency.update(counts.keys())

    return {term: math.log10(number_of_documents / df)
            for term, df in document_frequency.items()}


def tfidf_vector(counts, idf):
    """Build a unit-length TF-IDF vector from raw term counts.

    weight(t, d) = (1 + log10 tf(t, d)) * idf(t)    (sublinear tf scaling)
    The vector is then divided by its Euclidean length, so that long and
    short documents can be compared fairly (Manning et al., 2008, ch. 6).
    Terms unknown to the collection (idf missing) are ignored.
    """
    vector = {}
    for term, tf in counts.items():
        if term in idf and idf[term] > 0:
            vector[term] = (1 + math.log10(tf)) * idf[term]

    length = math.sqrt(sum(weight * weight for weight in vector.values()))
    if length == 0:
        return vector
    return {term: weight / length for term, weight in vector.items()}


# ---------------------------------------------------------------------------
# 4. COSINE SIMILARITY
# ---------------------------------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """Cosine of the angle between two unit vectors = their dot product.

    Only terms shared by both vectors contribute, so we loop over the
    smaller vector (the vectors are sparse dictionaries).
    """
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(weight * vector_b.get(term, 0.0) for term, weight in vector_a.items())


def similarity_matrix(document_vectors):
    """Cosine similarity for every pair of documents: {(i, j): score}."""
    doc_ids = sorted(document_vectors)
    return {(i, j): cosine_similarity(document_vectors[i], document_vectors[j])
            for i in doc_ids for j in doc_ids}


def rank_documents(query, document_vectors, idf, top_k=3):
    """Score every document against a free-text query, best first."""
    query_vector = tfidf_vector(Counter(preprocess_text(query)), idf)
    scores = [(doc_id, cosine_similarity(query_vector, vector))
              for doc_id, vector in document_vectors.items()]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# 5. OUTPUT HELPERS
# ---------------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_similarity_matrix(documents, matrix):
    doc_ids = sorted(documents)
    print("       " + "".join(f"  D{j:<4}" for j in doc_ids))
    for i in doc_ids:
        row = "".join(f"{matrix[(i, j)]:7.3f}" for j in doc_ids)
        print(f"  D{i:<3} {row}")


def save_heatmap(documents, matrix, file_path):
    """Draw the similarity matrix as a heatmap (skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    doc_ids = sorted(documents)
    labels = [f"D{i} {documents[i]['title']}" for i in doc_ids]
    values = [[matrix[(i, j)] for j in doc_ids] for i in doc_ids]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(doc_ids)), [f"D{i}" for i in doc_ids])
    ax.set_yticks(range(len(doc_ids)), labels)
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    ax.set_title("Cosine Similarity Between Documents (TF-IDF)")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return True


def verify_with_scikit_learn(documents, matrix):
    """Cross-check the hand-written results against scikit-learn.

    scikit-learn uses a slightly different idf formula (ln(N/df) + 1), so the
    numbers are not identical; what matters is that both agree on which
    document is most similar to each document.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    except ImportError:
        print("scikit-learn is not installed - verification skipped.")
        return

    doc_ids = sorted(documents)
    vectorizer = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True)
    sk_matrix = sk_cosine(vectorizer.fit_transform(documents[i]["text"] for i in doc_ids))

    agreements = 0
    for row, i in enumerate(doc_ids):
        ours = max((j for j in doc_ids if j != i), key=lambda j: matrix[(i, j)])
        theirs = doc_ids[max((c for c in range(len(doc_ids)) if c != row),
                             key=lambda c: sk_matrix[row][c])]
        agreements += ours == theirs
        print(f"  D{i}: nearest neighbour  ours = D{ours}   scikit-learn = D{theirs}")
    print(f"  Agreement on nearest neighbour: {agreements}/{len(doc_ids)} documents")


# ---------------------------------------------------------------------------
# 6. MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    documents = load_documents(DOCUMENTS_FOLDER)
    term_frequencies = compute_term_frequencies(documents)
    idf = compute_idf(term_frequencies)
    document_vectors = {doc_id: tfidf_vector(counts, idf)
                        for doc_id, counts in term_frequencies.items()}
    matrix = similarity_matrix(document_vectors)

    print_section("DOCUMENT COLLECTION")
    for doc_id, doc in documents.items():
        tokens = sum(term_frequencies[doc_id].values())
        print(f"  D{doc_id}  {doc['title']:<24} {doc['filename']:<34} "
              f"{tokens:>5} terms")
    print(f"\n  Vocabulary size (unique stemmed terms): {len(idf)}")

    print_section("TOP 5 TF-IDF TERMS PER DOCUMENT")
    for doc_id, vector in document_vectors.items():
        top_terms = sorted(vector.items(), key=lambda pair: pair[1], reverse=True)[:5]
        print(f"  D{doc_id}: " + ", ".join(f"{t} ({w:.3f})" for t, w in top_terms))

    print_section("COSINE SIMILARITY MATRIX")
    print_similarity_matrix(documents, matrix)

    print_section("MOST SIMILAR DOCUMENT PAIRS")
    pairs = sorted(((i, j) for (i, j) in matrix if i < j),
                   key=lambda pair: matrix[pair], reverse=True)
    for i, j in pairs[:5]:
        print(f"  {matrix[(i, j)]:.3f}  D{i} {documents[i]['title']:<24} "
              f"<-> D{j} {documents[j]['title']}")
    print("  ...")
    i, j = pairs[-1]
    print(f"  {matrix[(i, j)]:.3f}  D{i} {documents[i]['title']:<24} "
          f"<-> D{j} {documents[j]['title']}   (least similar)")

    print_section("QUERY RANKING")
    for query in SAMPLE_QUERIES + sys.argv[1:]:
        print(f'  Query: "{query}"')
        for rank, (doc_id, score) in enumerate(rank_documents(query, document_vectors, idf), 1):
            print(f"    {rank}. D{doc_id} {documents[doc_id]['title']:<24} score = {score:.4f}")
        print()

    print_section("VERIFICATION WITH SCIKIT-LEARN")
    verify_with_scikit_learn(documents, matrix)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    heatmap_path = os.path.join(OUTPUT_FOLDER, "similarity_heatmap.png")
    if save_heatmap(documents, matrix, heatmap_path):
        print(f"\n  Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    main()
