"""
Document ranking with Okapi BM25 and a Jelinek-Mercer language model
(TECH 400, Week 4).

This program:
  1. Reuses the Week 3 document loader and preprocessing (vsm_similarity.py).
  2. Collects the statistics both scoring functions need: term counts,
     document lengths, document frequencies and collection frequencies.
  3. Scores documents with Okapi BM25.
  4. Scores documents with a query likelihood language model using
     Jelinek-Mercer smoothing.
  5. Compares both against the Week 3 TF-IDF cosine ranking.
  6. Explores the parameters (k1, b and lambda) with small experiments.

The formulas are written by hand so each step is easy to follow; matplotlib
is only used to draw the parameter plots and is optional.

Usage:
    python ranking_functions.py                      # full report
    python ranking_functions.py "solar panels"       # also rank your own query
"""

import math
import os
import sys
from collections import Counter

from vsm_similarity import (DOCUMENTS_FOLDER, OUTPUT_FOLDER, compute_idf,
                            compute_term_frequencies, load_documents,
                            preprocess_text, rank_documents, tfidf_vector)

# Queries used for the comparison. The first four are the Week 3 queries;
# the last two are a one-word query and a long, verbose query.
SAMPLE_QUERIES = [
    "ranking documents with term weighting and cosine similarity",
    "neural networks learn from training data",
    "greenhouse gas emissions and solar power",
    "goalkeeper scores in the world cup",
    "crawler",
    "how do search engines crawl the web and index pages so that users "
    "can find relevant results quickly",
]

# Default parameters. k1 and b follow Robertson and Zaragoza (2009);
# the two lambda values follow Zhai and Lafferty (2004): about 0.1 for
# short keyword queries and about 0.7 for long queries.
BM25_K1 = 1.2
BM25_B = 0.75
JM_LAMBDA_SHORT = 0.1
JM_LAMBDA_LONG = 0.7


# ---------------------------------------------------------------------------
# 1. COLLECTION STATISTICS
# ---------------------------------------------------------------------------

class CollectionStatistics:
    """Everything BM25 and the language model need to know about the corpus.

    term_counts[d][t]   how often term t occurs in document d  (tf)
    doc_length[d]       number of terms in document d          (|d|)
    average_length      mean document length                   (avgdl)
    doc_frequency[t]    number of documents containing t       (df)
    collection_count[t] occurrences of t in the whole corpus   (cf)
    collection_length   total number of terms in the corpus    (|C|)
    """

    def __init__(self, term_counts):
        self.term_counts = term_counts
        self.doc_ids = sorted(term_counts)
        self.number_of_documents = len(term_counts)
        self.doc_length = {d: sum(c.values()) for d, c in term_counts.items()}
        self.average_length = sum(self.doc_length.values()) / self.number_of_documents

        self.doc_frequency = Counter()
        self.collection_count = Counter()
        for counts in term_counts.values():
            self.doc_frequency.update(counts.keys())
            self.collection_count.update(counts)
        self.collection_length = sum(self.collection_count.values())


# ---------------------------------------------------------------------------
# 2. OKAPI BM25
# ---------------------------------------------------------------------------

def bm25_idf(stats, term):
    """IDF used by BM25:  ln(1 + (N - df + 0.5) / (df + 0.5)).

    The "1 +" inside the log (the form used by Lucene) keeps the value
    positive. Without it, a term found in more than half of the documents
    would get a negative weight - see classic_rsj_idf below.
    """
    df = stats.doc_frequency[term]
    n = stats.number_of_documents
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def classic_rsj_idf(stats, term):
    """The original Robertson-Sparck Jones form:  ln((N - df + 0.5) / (df + 0.5))."""
    df = stats.doc_frequency[term]
    n = stats.number_of_documents
    return math.log((n - df + 0.5) / (df + 0.5))


def bm25_term_weight(tf, doc_length, average_length, k1=BM25_K1, b=BM25_B):
    """The saturating term frequency part of BM25.

                  tf * (k1 + 1)
        ----------------------------------------
        tf + k1 * (1 - b + b * |d| / avgdl)

    k1 controls how fast repeated occurrences stop adding score.
    b controls how strongly long documents are penalised (0 = not at all).
    """
    length_norm = 1 - b + b * doc_length / average_length
    return tf * (k1 + 1) / (tf + k1 * length_norm)


def bm25_breakdown(query_terms, stats, doc_id, k1=BM25_K1, b=BM25_B):
    """Per-term contributions to the BM25 score: [(term, tf, idf, part), ...]."""
    counts = stats.term_counts[doc_id]
    rows = []
    for term in query_terms:
        if stats.doc_frequency[term] == 0:
            continue  # term unknown to the collection
        tf = counts.get(term, 0)
        idf = bm25_idf(stats, term)
        weight = bm25_term_weight(tf, stats.doc_length[doc_id],
                                  stats.average_length, k1, b)
        rows.append((term, tf, idf, idf * weight))
    return rows


def bm25_score(query_terms, stats, doc_id, k1=BM25_K1, b=BM25_B):
    """BM25(q, d) = sum over query terms of idf(t) * term weight(t, d)."""
    return sum(part for *_, part in bm25_breakdown(query_terms, stats, doc_id, k1, b))


# ---------------------------------------------------------------------------
# 3. QUERY LIKELIHOOD WITH JELINEK-MERCER SMOOTHING
# ---------------------------------------------------------------------------

def jm_term_probability(term, stats, doc_id, lam):
    """Smoothed probability of a term under the document's language model.

        P(t | d) = (1 - lambda) * tf(t, d) / |d|  +  lambda * cf(t) / |C|

    The first part is the document model, the second the collection model.
    Because the collection part is never zero, a document is not ruled out
    just because it is missing one of the query terms.
    """
    document_part = stats.term_counts[doc_id].get(term, 0) / stats.doc_length[doc_id]
    collection_part = stats.collection_count[term] / stats.collection_length
    return (1 - lam) * document_part + lam * collection_part


def jm_breakdown(query_terms, stats, doc_id, lam):
    """Per-term contributions: [(term, tf, P(t|d), log P(t|d)), ...]."""
    rows = []
    for term in query_terms:
        if stats.collection_count[term] == 0:
            continue  # probability would be zero for every document
        p = jm_term_probability(term, stats, doc_id, lam)
        rows.append((term, stats.term_counts[doc_id].get(term, 0), p, math.log(p)))
    return rows


def jm_score(query_terms, stats, doc_id, lam=JM_LAMBDA_SHORT):
    """log P(q | d) = sum over query terms of log P(t | d).

    Logs are added instead of multiplying probabilities so that long
    queries do not underflow to zero. Scores are negative; closer to zero
    is better.
    """
    return sum(log_p for *_, log_p in jm_breakdown(query_terms, stats, doc_id, lam))


# ---------------------------------------------------------------------------
# 4. RANKING
# ---------------------------------------------------------------------------

def rank(score_function, query, stats, top_k=3, **parameters):
    """Score every document with the given function and return the best first."""
    query_terms = preprocess_text(query)
    scores = [(d, score_function(query_terms, stats, d, **parameters))
              for d in stats.doc_ids]
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


def print_ranking(label, ranking, documents, score_format="{:8.4f}"):
    print(f"    {label}")
    for position, (doc_id, score) in enumerate(ranking, 1):
        print(f"      {position}. D{doc_id} {documents[doc_id]['title']:<24} "
              f"score = {score_format.format(score)}")


def save_parameter_plots(stats, file_path):
    """Two plots: BM25 tf saturation, and the effect of lambda on JM scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) How much score each extra occurrence adds, in a document of
    #     average length. JM is shown on its own scale for comparison.
    tfs = list(range(0, 51))
    for k1, style in ((0.5, ":"), (1.2, "-"), (2.0, "--")):
        left.plot(tfs, [bm25_term_weight(tf, 1, 1, k1, 0.75) for tf in tfs],
                  style, label=f"BM25, k1 = {k1}")
    left.axhline(1.2 + 1, color="grey", linewidth=0.8)
    left.text(50, 2.25, "limit k1 + 1 = 2.2", ha="right", fontsize=8, color="grey")
    left.set_xlabel("Term frequency in the document (tf)")
    left.set_ylabel("BM25 term weight (before idf)")
    left.set_title("(a) BM25 term frequency saturation")
    left.legend(fontsize=8)

    # (b) Jelinek-Mercer scores for the long, verbose query as lambda moves
    #     from almost no smoothing to heavy smoothing.
    lambdas = [0.01] + [i / 20 for i in range(1, 20)]
    terms = preprocess_text(SAMPLE_QUERIES[5])
    for doc_id, style in ((2, "-"), (1, "--"), (3, ":"), (4, "-.")):
        right.plot(lambdas, [jm_score(terms, stats, doc_id, lam) for lam in lambdas],
                   style, label=f"D{doc_id}")
    right.set_xlabel("Smoothing parameter lambda")
    right.set_ylabel("log P(q | d)")
    right.set_title("(b) Jelinek-Mercer scores for the verbose query")
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 6. MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    documents = load_documents(DOCUMENTS_FOLDER)
    term_counts = compute_term_frequencies(documents)
    stats = CollectionStatistics(term_counts)

    # Week 3 TF-IDF vectors, used as the baseline.
    tfidf_idf = compute_idf(term_counts)
    tfidf_vectors = {d: tfidf_vector(c, tfidf_idf) for d, c in term_counts.items()}

    print_section("COLLECTION STATISTICS")
    for doc_id in stats.doc_ids:
        print(f"  D{doc_id}  {documents[doc_id]['title']:<24} |d| = "
              f"{stats.doc_length[doc_id]:>5}   (|d| / avgdl = "
              f"{stats.doc_length[doc_id] / stats.average_length:.2f})")
    print(f"\n  N = {stats.number_of_documents}   avgdl = {stats.average_length:.1f}"
          f"   |C| = {stats.collection_length}   vocabulary = {len(stats.doc_frequency)}")

    # -- Worked example: how each query term contributes ------------------
    example_query = "greenhouse gas emissions and solar power"
    example_terms = preprocess_text(example_query)
    print_section(f'WORKED EXAMPLE  query = "{example_query}"')
    print(f"  Query terms after preprocessing: {example_terms}\n")
    for doc_id in (6, 7):
        print(f"  D{doc_id} {documents[doc_id]['title']}  (|d| = {stats.doc_length[doc_id]})")
        print(f"    {'term':<10}{'tf':>5}{'df':>5}{'idf':>8}{'BM25':>9}"
              f"{'P(t|d)':>11}{'log P':>9}")
        jm_rows = jm_breakdown(example_terms, stats, doc_id, JM_LAMBDA_SHORT)
        for (term, tf, idf, part), (_, _, p, log_p) in zip(
                bm25_breakdown(example_terms, stats, doc_id), jm_rows):
            print(f"    {term:<10}{tf:>5}{stats.doc_frequency[term]:>5}{idf:>8.3f}"
                  f"{part:>9.3f}{p:>11.6f}{log_p:>9.3f}")
        print(f"    {'total':<28}{bm25_score(example_terms, stats, doc_id):>9.3f}"
              f"{'':>11}{jm_score(example_terms, stats, doc_id):>9.3f}\n")

    # -- Rankings under the three models -----------------------------------
    print_section("RANKINGS: TF-IDF COSINE vs BM25 vs JELINEK-MERCER")
    for query in SAMPLE_QUERIES + sys.argv[1:]:
        print(f'  Query: "{query}"')
        print_ranking("TF-IDF cosine (Week 3)",
                      rank_documents(query, tfidf_vectors, tfidf_idf), documents)
        print_ranking(f"BM25 (k1 = {BM25_K1}, b = {BM25_B})",
                      rank(bm25_score, query, stats), documents)
        print_ranking(f"LM Jelinek-Mercer (lambda = {JM_LAMBDA_SHORT})",
                      rank(jm_score, query, stats, lam=JM_LAMBDA_SHORT), documents)
        print()

    # -- BM25: term frequency saturation ------------------------------------
    print_section("BM25 TERM FREQUENCY SATURATION  (document of average length)")
    print(f"  {'tf':>5}{'raw tf':>9}{'1+log10 tf':>12}{'BM25 k1=1.2':>13}{'JM log P':>11}")
    for tf in (1, 2, 5, 10, 20, 50, 100):
        # JM: probability of a term with this tf in a document of average
        # length, for a term that is rare in the collection (cf = tf).
        p = ((1 - JM_LAMBDA_SHORT) * tf / stats.average_length
             + JM_LAMBDA_SHORT * tf / stats.collection_length)
        print(f"  {tf:>5}{tf:>9}{1 + math.log10(tf):>12.3f}"
              f"{bm25_term_weight(tf, 1, 1):>13.3f}{math.log(p):>11.3f}")

    # -- BM25: effect of the length normalisation parameter b ---------------
    query = "ranking documents with term weighting and cosine similarity"
    terms = preprocess_text(query)
    print_section("BM25 LENGTH NORMALISATION  (parameter b)")
    print(f'  Query: "{query}"')
    print(f"  {'b':>6}" + "".join(f"{'D' + str(d):>9}" for d in stats.doc_ids))
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  {b:>6.2f}" + "".join(f"{bm25_score(terms, stats, d, b=b):>9.3f}"
                                       for d in stats.doc_ids))

    # -- The negative IDF problem of the classic formula ---------------------
    print_section("CLASSIC RSJ IDF vs BM25 IDF (terms in more than half the documents)")
    common = sorted((t for t, df in stats.doc_frequency.items()
                     if df > stats.number_of_documents / 2),
                    key=lambda t: -stats.collection_count[t])
    print(f"  {len(common)} of {len(stats.doc_frequency)} terms occur in more than "
          f"{stats.number_of_documents // 2} of the {stats.number_of_documents} documents.")
    print(f"  {'term':<12}{'df':>4}{'classic idf':>14}{'BM25 idf':>11}")
    for term in ["data", "model", "search", "use", "energi"]:
        print(f"  {term:<12}{stats.doc_frequency[term]:>4}"
              f"{classic_rsj_idf(stats, term):>14.3f}{bm25_idf(stats, term):>11.3f}")

    # -- Jelinek-Mercer: effect of lambda -----------------------------------
    print_section("JELINEK-MERCER SMOOTHING  (parameter lambda)")
    for query in (SAMPLE_QUERIES[4], SAMPLE_QUERIES[5]):
        label = query if len(query) <= 60 else query[:57] + "..."
        print(f'  Query: "{label}"')
        for lam in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
            top = rank(jm_score, query, stats, top_k=3, lam=lam)
            print(f"    lambda = {lam:<6}" + "    ".join(
                f"D{d} {s:7.2f}" for d, s in top))
        print()

    # -- Missing query terms ------------------------------------------------
    query = "goalkeeper scores in the world cup"
    terms = preprocess_text(query)
    print_section("DOCUMENTS MISSING A QUERY TERM")
    print(f'  Query: "{query}"  terms = {terms}')
    for doc_id in stats.doc_ids:
        missing = [t for t in terms if term_counts[doc_id].get(t, 0) == 0]
        unsmoothed = "0 (log = -inf)" if missing else "> 0"
        print(f"  D{doc_id}  missing {str(missing):<40} unsmoothed P(q|d) = {unsmoothed:<15}"
              f" JM log P(q|d) = {jm_score(terms, stats, doc_id):8.2f}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plot_path = os.path.join(OUTPUT_FOLDER, "parameter_effects.png")
    if save_parameter_plots(stats, plot_path):
        print(f"\n  Parameter plots saved to {plot_path}")


if __name__ == "__main__":
    main()
