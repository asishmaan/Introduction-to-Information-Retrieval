"""
Passage retrieval on the Week 3 collection (TECH 400, Week 5).

Instead of returning whole documents, a passage retrieval system returns the
piece of text that answers the question. This program compares the main
ways of doing that:

  1. Passage segmentation (how the documents are cut up):
       - discourse passages : the paragraphs the author wrote
       - window passages    : fixed-size, overlapping windows of terms
       - sentence passages  : a few consecutive sentences
  2. Passage scoring:
       - BM25 over passages (each passage is treated as a small document)
       - term density       : the shortest span that covers the query terms
  3. Two-stage retrieval (the approach of Sarrouti & Ouatik El Alaoui, 2017):
     rank documents with BM25 first, then rank only the passages inside
     the top documents.
  4. Evidence combination: mix the passage score with the score of the
     document it comes from.

It reuses the Week 3 preprocessing and the Week 4 BM25 code.

Usage:
    python passage_retrieval.py                               # full report
    python passage_retrieval.py "what causes sea level rise"  # also your question
"""

import re
import sys
from collections import Counter

from ranking_functions import CollectionStatistics, bm25_score, print_section
from vsm_similarity import DOCUMENTS_FOLDER, load_documents, preprocess_text

# Questions used for the comparison: short factoid questions, the kind a
# question answering system sends to its passage retrieval step. Each one has
# an answer pattern; a passage counts as correct when it contains the answer,
# the way TREC question answering runs were judged (Tellex et al., 2003).
QUESTIONS = [
    # (question worded like the text, same question paraphrased, answer)
    ("which gas is the main driver of global warming",
     "what is the chief cause of the planet heating up", r"carbon dioxide"),
    ("who created Archie, the first search engine",
     "who built the earliest internet lookup tool", r"Emtage"),
    ("which algorithm did Google use to rank web pages",
     "how did Google order its results using links", r"PageRank"),
    ("who proposed the perceptron",
     "who invented the first trainable artificial neuron", r"Rosenblatt"),
    ("who formed the first large information retrieval research group",
     "who started the earliest big team studying document search", r"Salton"),
    ("how many players are on a football team",
     "how many people does each soccer side field", r"\b11\b|eleven"),
    ("what warming limit did nations agree to in the Paris Agreement",
     "what temperature cap did countries accept in the 2015 climate deal",
     r"2\s*\S?C"),
    ("who popularised backpropagation in 1986",
     "who made the error propagation training method widely known", r"Rumelhart"),
    ("who produced the first web robot",
     "who wrote the earliest program that automatically browsed websites",
     r"Matthew Gray"),
    ("which measures evaluate the results of an information retrieval system",
     "how is the quality of a search system's output judged",
     r"precision and recall"),
    ("which network won the ImageNet competition in 2012",
     "which model won the big image recognition contest in 2012", r"AlexNet"),
    ("which country is the largest producer of hydroelectricity",
     "which nation generates the most electricity from dams", r"China"),
    ("when were the Laws of the Game of football first issued",
     "in what year were soccer's rules first written down", r"1863"),
    ("which organisation determines the Laws of the Game",
     "which body is in charge of the rules of soccer",
     r"IFAB|International Football Association Board"),
    ("who proposed the classic vector space model",
     "who came up with representing texts as weighted term vectors",
     r"Salton, Wong and Yang"),
]

# Questions printed in full with the top passage of every method.
SHOWCASE = [0, 1, 5, 14]

WINDOW_SIZE = 50        # terms per window passage (Callan, 1994, used 50-200)
WINDOW_STRIDE = 25      # windows overlap by half, so no answer is cut in two
SENTENCES_PER_PASSAGE = 3
TOP_DOCUMENTS = 2       # documents kept by the first stage of two-stage retrieval
DOC_WEIGHT = 0.3        # weight of the document score in evidence combination


# ---------------------------------------------------------------------------
# 1. PASSAGE SEGMENTATION
# ---------------------------------------------------------------------------

def paragraph_passages(text):
    """Discourse passages: every non-empty line of the file is a paragraph.

    The first line is the title, so it is skipped.
    """
    lines = [line.strip() for line in text.splitlines()[1:]]
    return [line for line in lines if len(line.split()) >= 5]


def split_sentences(text):
    """A simple sentence splitter: break after . ! ? followed by a capital."""
    body = " ".join(text.splitlines()[1:])
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body) if s.strip()]


def sentence_passages(text, size=SENTENCES_PER_PASSAGE):
    """Sentence passages: groups of `size` consecutive sentences, sliding by one."""
    sentences = split_sentences(text)
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i + size]) for i in range(len(sentences) - size + 1)]


def window_passages(text, size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Window passages: `size` words, starting every `stride` words.

    Windows ignore the document structure, so a window can start or end in
    the middle of a sentence. Overlapping them by half means that any span
    shorter than `stride` words is fully inside at least one window.
    """
    words = " ".join(text.splitlines()[1:]).split()
    starts = range(0, max(len(words) - size, 0) + 1, stride)
    return [" ".join(words[s:s + size]) for s in starts]


SEGMENTERS = {
    "paragraph": paragraph_passages,
    "window": window_passages,
    "sentence": sentence_passages,
}


class PassageIndex:
    """All passages of the collection produced by one segmenter.

    Each passage gets the ID (doc_id, passage_number), so its source document
    is always known. BM25 statistics (df, avgdl ...) are computed over the
    passages, not the documents: a passage is scored as a small document.
    """

    def __init__(self, documents, segmenter):
        self.text = {}
        for doc_id, document in documents.items():
            for number, passage in enumerate(segmenter(document["text"])):
                self.text[(doc_id, number)] = passage
        self.terms = {pid: preprocess_text(t) for pid, t in self.text.items()}
        self.stats = CollectionStatistics(
            {pid: Counter(terms) for pid, terms in self.terms.items()})


# ---------------------------------------------------------------------------
# 2. PASSAGE SCORING
# ---------------------------------------------------------------------------

def bm25_passages(question, index, candidates=None):
    """Rank passages with BM25. `candidates` limits the passages scored."""
    terms = preprocess_text(question)
    pids = candidates if candidates is not None else index.stats.doc_ids
    scores = [(pid, bm25_score(terms, index.stats, pid)) for pid in pids]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def density_score(query_terms, passage_terms):
    """Term density: reward passages where the query terms occur close together.

    Finds the shortest span of the passage that contains as many distinct
    query terms as the passage has (the cover), then scores

        matched terms ** 2 / span length

    so more matched terms and a tighter span both raise the score. This is
    the idea behind the MultiText cover density ranking (Clarke et al., 2000)
    and the density passage retrievers compared by Tellex et al. (2003).
    """
    wanted = set(query_terms)
    present = wanted & set(passage_terms)
    if not present:
        return 0.0

    best_span = len(passage_terms)
    window_counts = {}
    left = 0
    covered = 0
    for right, term in enumerate(passage_terms):
        if term in present:
            window_counts[term] = window_counts.get(term, 0) + 1
            if window_counts[term] == 1:
                covered += 1
        while covered == len(present):
            best_span = min(best_span, right - left + 1)
            left_term = passage_terms[left]
            if left_term in present:
                window_counts[left_term] -= 1
                if window_counts[left_term] == 0:
                    covered -= 1
            left += 1
    return len(present) ** 2 / best_span


def density_passages(question, index):
    terms = preprocess_text(question)
    scores = [(pid, density_score(terms, index.terms[pid])) for pid in index.text]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


# ---------------------------------------------------------------------------
# 3. TWO-STAGE RETRIEVAL AND EVIDENCE COMBINATION
# ---------------------------------------------------------------------------

def two_stage(question, index, doc_stats, top_documents=TOP_DOCUMENTS):
    """Stage 1: BM25 over whole documents. Stage 2: BM25 over their passages.

    This is how Sarrouti and Ouatik El Alaoui (2017) organise biomedical
    question answering: the document retriever narrows the search, and the
    passage retriever only looks inside the documents that survived.
    """
    terms = preprocess_text(question)
    doc_ranking = sorted(((d, bm25_score(terms, doc_stats, d)) for d in doc_stats.doc_ids),
                         key=lambda pair: pair[1], reverse=True)
    kept = {d for d, _ in doc_ranking[:top_documents]}
    candidates = [pid for pid in index.stats.doc_ids if pid[0] in kept]
    return bm25_passages(question, index, candidates)


def combined(question, index, doc_stats, weight=DOC_WEIGHT):
    """Evidence combination: (1 - w) * passage score + w * document score.

    Both scores are divided by their maximum first so they are on the same
    scale. A passage from a document that is about the question as a whole
    gets a boost over an isolated passage that only shares a few words.
    """
    terms = preprocess_text(question)
    doc_scores = {d: bm25_score(terms, doc_stats, d) for d in doc_stats.doc_ids}
    passage_scores = dict(bm25_passages(question, index))
    max_doc = max(doc_scores.values()) or 1
    max_passage = max(passage_scores.values()) or 1
    scores = [(pid, (1 - weight) * s / max_passage + weight * doc_scores[pid[0]] / max_doc)
              for pid, s in passage_scores.items()]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


# ---------------------------------------------------------------------------
# 4. MAIN PROGRAM
# ---------------------------------------------------------------------------

def shorten(text, limit=150):
    return text if len(text) <= limit else text[:limit - 3] + "..."


def print_top(label, ranking, index, documents, top_k=1):
    print(f"    {label}")
    for pid, score in ranking[:top_k]:
        doc_id, number = pid
        print(f"      D{doc_id}.{number:<3} {documents[doc_id]['title']:<24} score = {score:.3f}")
        print(f"        \"{shorten(index.text[pid])}\"")


def first_correct_rank(ranking, index, pattern, depth=5):
    """Rank (1..depth) of the first passage containing the answer, or None."""
    for position, (pid, score) in enumerate(ranking[:depth], 1):
        if score > 0 and re.search(pattern, index.text[pid], re.IGNORECASE):
            return position
    return None


def main():
    documents = load_documents(DOCUMENTS_FOLDER)
    doc_stats = CollectionStatistics(
        {d: Counter(preprocess_text(doc["text"])) for d, doc in documents.items()})
    indexes = {name: PassageIndex(documents, seg) for name, seg in SEGMENTERS.items()}
    sentences = indexes["sentence"]

    # Every method: (label, passage index it uses, ranking function).
    methods = [(f"BM25, {name} passages", index,
                lambda q, index=index: bm25_passages(q, index))
               for name, index in indexes.items()]
    methods += [
        ("Term density, sentence passages", sentences,
         lambda q: density_passages(q, sentences)),
        (f"Two-stage (top {TOP_DOCUMENTS} docs), sentences", sentences,
         lambda q: two_stage(q, sentences, doc_stats)),
        (f"Passage + document (w = {DOC_WEIGHT}), sentences", sentences,
         lambda q: combined(q, sentences, doc_stats)),
    ]

    print_section("PASSAGE SEGMENTATION")
    print(f"  {'segmenter':<12}{'passages':>10}{'avg terms':>11}{'min':>6}{'max':>6}")
    for name, index in indexes.items():
        lengths = list(index.stats.doc_length.values())
        print(f"  {name:<12}{len(lengths):>10}{index.stats.average_length:>11.1f}"
              f"{min(lengths):>6}{max(lengths):>6}")
    print(f"\n  Whole documents: {doc_stats.number_of_documents}, "
          f"avg terms = {doc_stats.average_length:.1f}")

    showcase = ([QUESTIONS[i][0] for i in SHOWCASE] + [QUESTIONS[i][1] for i in SHOWCASE]
                + sys.argv[1:])
    for question in showcase:
        terms = preprocess_text(question)
        print_section(f'QUESTION: "{question}"')
        print(f"  Terms: {terms}")
        best_doc = max(doc_stats.doc_ids, key=lambda d: bm25_score(terms, doc_stats, d))
        print(f"  Best whole document (BM25): D{best_doc} {documents[best_doc]['title']}"
              f" ({doc_stats.doc_length[best_doc]} terms to read)\n")
        for label, index, ranker in methods:
            print_top(label, ranker(question), index, documents)

    print_section("EVALUATION  (does the passage contain the answer?)")
    print(f"  {len(QUESTIONS)} factoid questions, asked twice: in the words of the")
    print("  text, then paraphrased. Top-1 = answer in the first passage;")
    print("  MRR@5 = mean of 1/rank of the first correct passage in the top 5.")
    for column, title in ((0, "WORDED LIKE THE TEXT"), (1, "PARAPHRASED")):
        print(f"\n  {title}")
        print(f"  {'method':<42}{'top-1':>7}{'top-5':>7}{'MRR@5':>8}{'words':>7}")
        for label, index, ranker in methods:
            ranks, words = [], []
            for entry in QUESTIONS:
                ranking = ranker(entry[column])
                ranks.append(first_correct_rank(ranking, index, entry[2]))
                words.append(len(index.text[ranking[0][0]].split()))
            top1 = sum(r == 1 for r in ranks)
            top5 = sum(r is not None for r in ranks)
            mrr = sum(1 / r for r in ranks if r) / len(ranks)
            print(f"  {label:<42}{top1:>4}/{len(ranks):<2}{top5:>4}/{len(ranks):<2}"
                  f"{mrr:>8.3f}{sum(words) / len(words):>7.0f}")
    print("\n  words = average length of the top passage the user has to read.")

    print_section("PARAPHRASED QUESTIONS: RANK OF THE FIRST CORRECT PASSAGE (- = not in top 5)")
    print(f"  {'question':<52}" + "".join(f"{'M' + str(i + 1):>4}" for i in range(len(methods))))
    for _, question, pattern in QUESTIONS:
        cells = []
        for label, index, ranker in methods:
            r = first_correct_rank(ranker(question), index, pattern)
            cells.append(f"{r if r else '-':>4}")
        print(f"  {shorten(question, 50):<52}" + "".join(cells))
    print()
    for i, (label, _, _) in enumerate(methods, 1):
        print(f"  M{i} = {label}")


if __name__ == "__main__":
    main()
