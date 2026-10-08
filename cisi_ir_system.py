"""
A complete information retrieval system for the CISI test collection
(TECH 400, Week 6).

The earlier weeks each built one piece of a search engine on a hand-made
collection of eight articles. This program puts the pieces together into one
system and runs it on a real, publicly available test collection: CISI, the
1,460 abstracts and 112 queries released by the Centre for Inventions and
Scientific Information and distributed with the SMART system.

The architecture has five stages:

  1. Dataset     parse CISI.ALL, CISI.QRY and CISI.REL into documents,
                 queries and relevance judgements.
  2. Preprocess  lowercase, tokenize, remove stop words, Porter stemming
                 (reused from Week 3). The effect of each step on the
                 vocabulary is measured.
  3. Index       build an inverted index: for every term, the postings list
                 of (document, term frequency) pairs, plus the document
                 lengths, document frequencies and collection frequencies
                 the ranking functions need.
  4. Retrieve    four models score the documents, all of them reading the
                 same postings lists:
                   - Boolean OR      (Week 2)
                   - TF-IDF cosine   (Week 3)
                   - Okapi BM25      (Week 4)
                   - query likelihood with Jelinek-Mercer smoothing (Week 4)
  5. Query       term-at-a-time scoring with accumulators: only documents
                 that contain at least one query term are scored, so most of
                 the collection is never touched.

A short effectiveness check (precision@10, recall@10, R-precision) confirms
the system works. The full evaluation - MAP, nDCG, precision-recall curves -
is the Week 7 assignment.

Usage:
    python cisi_ir_system.py                          # full report
    python cisi_ir_system.py "automatic indexing"     # also run your query
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict

from ranking_functions import (BM25_B, BM25_K1, CollectionStatistics, bm25_idf,
                               bm25_term_weight, jm_term_probability)
from vsm_similarity import OUTPUT_FOLDER, STOP_WORDS, preprocess_text

# Folder holding CISI.ALL, CISI.QRY and CISI.REL.
CISI_FOLDER = os.path.join("data", "cisi")

# Jelinek-Mercer smoothing. CISI queries are verbose, and Zhai and Lafferty
# (2004) found lambda near 0.7 worked best for long queries.
JM_LAMBDA = 0.7

# Queries shown in full in the report, picked to be short, average and long.
SHOWCASE_QUERIES = [3, 18, 35]

TOP_K = 10


# ---------------------------------------------------------------------------
# 1. THE DATASET
# ---------------------------------------------------------------------------

def parse_cisi_file(path):
    """Split a CISI file into records keyed by the number after ".I".

    Both CISI.ALL and CISI.QRY use the same SMART-style tagged format:

        .I 1            record number
        .T              title
        .A              author(s)
        .W              the abstract or the query text
        .B              bibliographic note
        .X              citations (document number, type, document number)

    A record is returned as {"T": "...", "A": "...", "W": "...", ...}: the
    lines under each tag, joined with spaces.
    """
    records = {}
    record_id = None
    tag = None

    with open(path, "r", encoding="latin-1") as opened_file:
        for line in opened_file:
            line = line.rstrip("\n")
            if line.startswith(".I"):
                record_id = int(line.split()[1])
                records[record_id] = defaultdict(list)
                tag = None
            elif re.fullmatch(r"\.[A-Z]\s*", line):
                tag = line.strip()[1]
            elif record_id is not None and tag is not None:
                records[record_id][tag].append(line.strip())

    return {rid: {t: " ".join(lines).strip() for t, lines in fields.items()}
            for rid, fields in records.items()}


def load_cisi_documents(folder=CISI_FOLDER):
    """Load the 1,460 CISI abstracts.

    The searchable text of a document is its title plus its abstract. The
    author field is kept for display but not indexed: an author's name is
    not what these queries ask about. The .X citation block is numeric and
    would be dropped by the tokenizer anyway.
    """
    documents = {}
    for doc_id, fields in parse_cisi_file(os.path.join(folder, "CISI.ALL")).items():
        title = fields.get("T", "")
        abstract = fields.get("W", "")
        documents[doc_id] = {"title": title,
                             "author": fields.get("A", ""),
                             "text": f"{title} {abstract}"}
    return documents


def load_cisi_queries(folder=CISI_FOLDER):
    """Load the 112 CISI queries.

    Queries 1-76 are natural language questions typed by users. Queries
    77-112 are whole abstracts used as queries ("more like this" searches),
    so they are far longer. Only the .W field is used for both.
    """
    return {qid: fields.get("W", "")
            for qid, fields in parse_cisi_file(os.path.join(folder, "CISI.QRY")).items()}


def load_cisi_relevance(folder=CISI_FOLDER):
    """Load the relevance judgements: {query_id: {relevant document ids}}.

    Each line of CISI.REL is "query_id document_id 0 0.000000". The last two
    columns are unused in this release, so relevance is binary: a document
    is either listed for a query or it is not. Judgements exist for 76 of
    the 112 queries; the rest are unjudged and cannot be scored.
    """
    relevance = defaultdict(set)
    with open(os.path.join(folder, "CISI.REL"), "r", encoding="latin-1") as opened_file:
        for line in opened_file:
            parts = line.split()
            if len(parts) >= 2:
                relevance[int(parts[0])].add(int(parts[1]))
    return dict(relevance)


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
# The indexing pipeline uses preprocess_text from Week 3 (lowercase, keep
# alphabetic words, drop stop words, Porter stemming). The two functions
# below are the same pipeline with the later steps switched off, so the
# report can show what each step costs and saves.

def tokens_only(text):
    """Steps 1 and 2 only: lowercase and keep alphabetic words of length > 1."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 1]


def tokens_without_stop_words(text):
    """Steps 1-3: tokenize, then drop stop words."""
    return [t for t in tokens_only(text) if t not in STOP_WORDS]


PIPELINES = [
    ("tokenize only", tokens_only),
    ("+ stop word removal", tokens_without_stop_words),
    ("+ Porter stemming", preprocess_text),
]


# ---------------------------------------------------------------------------
# 3. INDEXING
# ---------------------------------------------------------------------------

class InvertedIndex:
    """An inverted index over the collection, plus the statistics for ranking.

    postings[t]   list of (document id, tf) pairs, sorted by document id
    stats         CollectionStatistics from Week 4: document lengths, df,
                  collection frequencies and the collection length
    idf[t]        log10(N / df(t)), the Week 3 weighting
    norm[d]       Euclidean length of document d's TF-IDF vector, stored so
                  cosine normalization is not recomputed on every query

    Storing tf inside the postings list is what makes term-at-a-time scoring
    possible: one pass over a term's postings updates every document that
    contains it (Manning et al., 2008, ch. 7).
    """

    def __init__(self, term_counts):
        self.stats = CollectionStatistics(term_counts)

        self.postings = defaultdict(list)
        for doc_id in self.stats.doc_ids:
            for term, tf in term_counts[doc_id].items():
                self.postings[term].append((doc_id, tf))

        n = self.stats.number_of_documents
        self.idf = {term: math.log10(n / df)
                    for term, df in self.stats.doc_frequency.items()}

        self.norm = {}
        for doc_id, counts in term_counts.items():
            weights = [(1 + math.log10(tf)) * self.idf[term]
                       for term, tf in counts.items()]
            self.norm[doc_id] = math.sqrt(sum(w * w for w in weights)) or 1.0

    @property
    def vocabulary_size(self):
        return len(self.postings)

    @property
    def number_of_postings(self):
        return sum(len(postings) for postings in self.postings.values())

    def query_terms(self, query):
        """Preprocess a query and keep only the terms the index knows."""
        return [t for t in preprocess_text(query) if t in self.postings]


def build_index(documents):
    """Preprocess every document and build the inverted index."""
    term_counts = {doc_id: Counter(preprocess_text(doc["text"]))
                   for doc_id, doc in documents.items()}
    # A document whose text preprocesses to nothing would divide by zero in
    # the length normalization, so give it one placeholder term.
    for counts in term_counts.values():
        if not counts:
            counts["emptydocument"] = 1
    return InvertedIndex(term_counts)


# ---------------------------------------------------------------------------
# 4. RETRIEVAL MODELS
# ---------------------------------------------------------------------------
# Every model has the same shape: walk the postings list of each query term
# and add that term's contribution to an accumulator per document. The models
# differ only in what they add.

def score_boolean_or(query_terms, index):
    """Baseline from Week 2: count how many query terms a document contains.

    This is Boolean OR with the matches counted, the simplest thing that can
    rank at all. It ignores how often a term occurs, how rare it is and how
    long the document is, which is exactly what the other three models fix.
    """
    scores = defaultdict(float)
    for term in sorted(set(query_terms)):
        for doc_id, _ in index.postings[term]:
            scores[doc_id] += 1.0
    return scores


def score_tfidf_cosine(query_terms, index):
    """Week 3: cosine similarity between TF-IDF vectors.

        weight(t, d) = (1 + log10 tf(t, d)) * idf(t)

    Document weights are divided by the stored vector length and the query
    weights by the query's own length, so the result is the cosine of the
    angle between the two vectors.
    """
    query_counts = Counter(query_terms)
    query_weights = {t: (1 + math.log10(tf)) * index.idf[t]
                     for t, tf in query_counts.items()}
    query_norm = math.sqrt(sum(w * w for w in query_weights.values())) or 1.0

    scores = defaultdict(float)
    for term, query_weight in query_weights.items():
        for doc_id, tf in index.postings[term]:
            doc_weight = (1 + math.log10(tf)) * index.idf[term]
            scores[doc_id] += query_weight * doc_weight / (query_norm * index.norm[doc_id])
    return scores


def score_bm25(query_terms, index, k1=BM25_K1, b=BM25_B):
    """Week 4: Okapi BM25, reusing bm25_idf and bm25_term_weight.

    Repeated occurrences saturate (controlled by k1) and long documents are
    discounted (controlled by b). CISI abstracts vary widely in length, so
    the length normalization matters more here than it did on the eight
    hand-picked articles of Week 3.
    """
    stats = index.stats
    scores = defaultdict(float)
    for term in sorted(set(query_terms)):
        idf = bm25_idf(stats, term)
        for doc_id, tf in index.postings[term]:
            scores[doc_id] += idf * bm25_term_weight(
                tf, stats.doc_length[doc_id], stats.average_length, k1, b)
    return scores


def score_jelinek_mercer(query_terms, index, lam=JM_LAMBDA):
    """Week 4: query likelihood with Jelinek-Mercer smoothing.

        log P(q | d) = sum over query terms of
                       log[(1 - lambda) tf(t, d)/|d| + lambda cf(t)/|C|]

    Every document gets a non-zero probability from the collection model, so
    a document is not ruled out for missing one query term. Walking only the
    postings lists would leave those documents unscored, so each document
    starts at the score it would get from the collection model alone and the
    postings add the difference. The ranking is identical to scoring all
    1,460 documents directly, but only the postings are read.
    """
    stats = index.stats
    collection_only = {}
    for term in sorted(set(query_terms)):
        probability = lam * stats.collection_count[term] / stats.collection_length
        collection_only[term] = math.log(probability)

    baseline = sum(collection_only[term] for term in query_terms)
    scores = {doc_id: baseline for doc_id in stats.doc_ids}

    for term in query_terms:
        for doc_id, _ in index.postings[term]:
            smoothed = jm_term_probability(term, stats, doc_id, lam)
            scores[doc_id] += math.log(smoothed) - collection_only[term]
    return scores


MODELS = [
    ("Boolean OR (term match count)", score_boolean_or),
    ("TF-IDF cosine", score_tfidf_cosine),
    (f"Okapi BM25 (k1={BM25_K1}, b={BM25_B})", score_bm25),
    (f"Jelinek-Mercer LM (lambda={JM_LAMBDA})", score_jelinek_mercer),
]


# ---------------------------------------------------------------------------
# 5. QUERY PROCESSING
# ---------------------------------------------------------------------------

def search(query, index, score_function, top_k=TOP_K):
    """Run one query: preprocess, score, sort, return the top k.

    Returns (ranking, documents scored). The second value shows how much of
    the collection the accumulators avoided touching.
    """
    terms = index.query_terms(query)
    if not terms:
        return [], 0
    scores = score_function(terms, index)
    # Sorting on (-score, document id) breaks ties the same way on every run,
    # so the report and the saved output always agree.
    ranking = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranking[:top_k], len(scores)


# ---------------------------------------------------------------------------
# 6. EFFECTIVENESS CHECK
# ---------------------------------------------------------------------------
# A sanity check that the ranking is doing something real. The full
# evaluation - MAP, nDCG and precision-recall curves - is Week 7.

def precision_at_k(ranked_ids, relevant, k):
    if k == 0:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / len(relevant)


def r_precision(ranked_ids, relevant):
    """Precision at rank R, where R is the number of relevant documents.

    This makes queries with 5 and 50 relevant documents comparable, which
    matters on CISI: the number of relevant documents per query ranges from
    a handful to over a hundred.
    """
    r = len(relevant)
    if r == 0:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:r]) / r


def evaluate(index, score_function, queries, relevance, k=TOP_K):
    """Average precision@k, recall@k and R-precision over the judged queries."""
    judged = sorted(qid for qid in relevance if qid in queries)
    rows = []
    for qid in judged:
        depth = max(k, len(relevance[qid]))
        ranking, _ = search(queries[qid], index, score_function, top_k=depth)
        ranked_ids = [doc_id for doc_id, _ in ranking]
        rows.append((precision_at_k(ranked_ids, relevance[qid], k),
                     recall_at_k(ranked_ids, relevance[qid], k),
                     r_precision(ranked_ids, relevance[qid])))
    return [sum(column) / len(rows) for column in zip(*rows)], len(rows)


# ---------------------------------------------------------------------------
# 7. OUTPUT HELPERS
# ---------------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def shorten(text, limit=58):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def print_ranking(label, ranking, documents, relevant=None):
    print(f"    {label}")
    for position, (doc_id, score) in enumerate(ranking, 1):
        mark = ""
        if relevant:
            mark = "  <- relevant" if doc_id in relevant else ""
        print(f"      {position:>2}. D{doc_id:<5}"
              f"{shorten(documents[doc_id]['title']):<58}{score:>9.3f}{mark}")


def save_figure(index, file_path):
    """Two plots: the term distribution (Zipf) and the document length spread."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Collection frequency against rank on log-log axes. A near-straight
    #     line means the vocabulary follows Zipf's law, which is why a few
    #     terms dominate the postings and why idf is needed at all.
    frequencies = sorted(index.stats.collection_count.values(), reverse=True)
    left.loglog(range(1, len(frequencies) + 1), frequencies)
    left.set_xlabel("Term rank (most frequent first)")
    left.set_ylabel("Collection frequency")
    left.set_title("(a) Zipf's law in the CISI vocabulary")
    left.grid(True, which="both", linewidth=0.3)

    # (b) How long the indexed documents are. BM25's b parameter and the
    #     language model both normalize by this length.
    lengths = list(index.stats.doc_length.values())
    right.hist(lengths, bins=40, color="steelblue", edgecolor="white")
    right.axvline(index.stats.average_length, color="firebrick", linestyle="--",
                  label=f"mean = {index.stats.average_length:.0f}")
    right.set_xlabel("Indexed terms per document")
    right.set_ylabel("Number of documents")
    right.set_title("(b) Document length distribution")
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 8. MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    documents = load_cisi_documents()
    queries = load_cisi_queries()
    relevance = load_cisi_relevance()

    print_section("1. THE DATASET: CISI")
    judged = sorted(qid for qid in relevance if qid in queries)
    relevant_counts = [len(relevance[qid]) for qid in judged]
    print(f"  Source                    {CISI_FOLDER} (CISI.ALL, CISI.QRY, CISI.REL)")
    print(f"  Documents                 {len(documents)}")
    print(f"  Queries                   {len(queries)}")
    print(f"  Queries with judgements   {len(judged)}")
    print(f"  Relevance judgements      {sum(relevant_counts)} (binary)")
    print(f"  Relevant per query        min {min(relevant_counts)}, "
          f"mean {sum(relevant_counts) / len(relevant_counts):.1f}, "
          f"max {max(relevant_counts)}")

    print("\n  Sample document")
    print(f"    D1 title    {documents[1]['title']}")
    print(f"       author   {documents[1]['author']}")
    print(f"       text     {shorten(documents[1]['text'], 66)}")
    print("\n  Sample queries")
    for qid in (1, 3, 77):
        print(f"    Q{qid:<4} {shorten(queries[qid], 66)}")

    print_section("2. PREPROCESSING: WHAT EACH STEP REMOVES")
    print(f"  {'pipeline':<24}{'tokens':>12}{'vocabulary':>13}{'vs. step 1':>13}")
    first_vocabulary = None
    for label, pipeline in PIPELINES:
        tokens = 0
        vocabulary = set()
        for doc in documents.values():
            terms = pipeline(doc["text"])
            tokens += len(terms)
            vocabulary.update(terms)
        if first_vocabulary is None:
            first_vocabulary = len(vocabulary)
        reduction = 100 * (1 - len(vocabulary) / first_vocabulary)
        print(f"  {label:<24}{tokens:>12,}{len(vocabulary):>13,}{reduction:>12.1f}%")
    print("\n  Stop word removal cuts tokens but barely touches the vocabulary;")
    print("  stemming is the opposite - it merges word forms into fewer terms.")

    index = build_index(documents)
    stats = index.stats
    once = sum(df == 1 for df in stats.doc_frequency.values())

    print_section("3. THE INVERTED INDEX")
    print(f"  Documents indexed         {stats.number_of_documents}")
    print(f"  Vocabulary (index terms)  {index.vocabulary_size:,}")
    print(f"  Postings                  {index.number_of_postings:,}")
    print(f"  Tokens indexed            {stats.collection_length:,}")
    print(f"  Average document length   {stats.average_length:.1f} terms")
    print(f"  Shortest / longest        {min(stats.doc_length.values())} / "
          f"{max(stats.doc_length.values())} terms")
    print(f"  Terms in only one document {once:,} "
          f"({100 * once / index.vocabulary_size:.0f}% of the vocabulary)")

    print("\n  Ten most frequent index terms")
    print(f"  {'term':<14}{'cf':>8}{'df':>8}{'idf':>9}")
    for term, cf in stats.collection_count.most_common(10):
        print(f"  {term:<14}{cf:>8}{stats.doc_frequency[term]:>8}{index.idf[term]:>9.3f}")

    example = "classif"
    print(f'\n  Postings list for "{example}" '
          f"(first 10 of {len(index.postings[example])} entries)")
    print("    " + "  ".join(f"D{d}:{tf}" for d, tf in index.postings[example][:10]))

    print_section("4. RETRIEVAL ON THREE SHOWCASE QUERIES")
    for qid in SHOWCASE_QUERIES:
        terms = index.query_terms(queries[qid])
        print(f'\n  Q{qid}: "{shorten(queries[qid], 70)}"')
        print(f"  Index terms ({len(terms)}): {terms[:10]}"
              f"{' ...' if len(terms) > 10 else ''}")
        print(f"  Relevant documents in the ground truth: {len(relevance.get(qid, []))}\n")
        for label, score_function in MODELS:
            ranking, scored = search(queries[qid], index, score_function, top_k=5)
            print_ranking(f"{label}   [{scored} documents scored]",
                          ranking, documents, relevance.get(qid, set()))
            print()

    print_section("5. QUERY PROCESSING COST")
    print("  Accumulators only touch documents that hold a query term, so the")
    print("  rest of the collection is never scored. The smoothed language")
    print("  model is the exception: every document gets a non-zero score, so")
    print("  all 1,460 have to be ranked however short the query is.")
    print(f"\n  {'query':<9}{'terms':>7}{'TF-IDF/BM25':>13}{'of all':>9}{'JM LM':>8}")
    for qid in SHOWCASE_QUERIES + [77]:
        terms = index.query_terms(queries[qid])
        _, scored = search(queries[qid], index, score_tfidf_cosine)
        _, jm_scored = search(queries[qid], index, score_jelinek_mercer)
        print(f"  Q{qid:<8}{len(terms):>7}{scored:>13}"
              f"{100 * scored / stats.number_of_documents:>8.0f}%{jm_scored:>8}")
    print("\n  Q77 is a whole abstract used as a query, so it reaches almost")
    print("  every document; the short questions reach far fewer.")

    print_section(f"6. EFFECTIVENESS CHECK ON THE {len(judged)} JUDGED QUERIES")
    print(f"  {'model':<34}{'P@10':>9}{'R@10':>9}{'R-prec':>9}")
    count = 0
    for label, score_function in MODELS:
        (p10, r10, rprec), count = evaluate(index, score_function, queries, relevance)
        print(f"  {label:<34}{p10:>9.3f}{r10:>9.3f}{rprec:>9.3f}")
    print(f"\n  Averaged over {count} queries. The full evaluation (MAP, nDCG,")
    print("  precision-recall curves) follows in the Week 7 assignment.")

    print_section("7. WHY THESE PARAMETERS (MODEL SELECTION)")
    print("  BM25: b controls the length penalty. CISI abstracts are short and")
    print("  fairly even in length, so the default 0.75 is not critical.")
    print(f"\n  {'setting':<34}{'P@10':>9}{'R-prec':>9}")
    for b in (0.0, 0.5, 0.75, 1.0):
        (p10, _, rprec), _ = evaluate(
            index, lambda terms, idx, b=b: score_bm25(terms, idx, b=b),
            queries, relevance)
        print(f"  {'BM25, b = ' + str(b):<34}{p10:>9.3f}{rprec:>9.3f}")
    print("\n  Language model: lambda is the weight on the collection model.")
    print("  Zhai and Lafferty (2004) report that long queries need heavy")
    print("  smoothing. CISI queries are long and the sweep agrees: P@10 is")
    print("  flat from 0.5 to 0.7, and R-precision keeps climbing past it.")
    for lam in (0.1, 0.3, 0.5, 0.7, 0.9):
        (p10, _, rprec), _ = evaluate(
            index, lambda terms, idx, lam=lam: score_jelinek_mercer(terms, idx, lam),
            queries, relevance)
        print(f"  {'JM LM, lambda = ' + str(lam):<34}{p10:>9.3f}{rprec:>9.3f}")

    if len(sys.argv) > 1:
        print_section("8. YOUR QUERIES")
        for query in sys.argv[1:]:
            print(f'\n  Query: "{query}"')
            print(f"  Index terms: {index.query_terms(query)}\n")
            for label, score_function in MODELS[1:]:
                ranking, _ = search(query, index, score_function, top_k=5)
                print_ranking(label, ranking, documents)
                print()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    figure_path = os.path.join(OUTPUT_FOLDER, "cisi_collection.png")
    if save_figure(index, figure_path):
        print(f"\n  Figure saved to {figure_path}")


if __name__ == "__main__":
    main()
