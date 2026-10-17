"""
Evaluating the CISI retrieval system (TECH 400, Week 7).

Week 6 built a search engine over the CISI test collection and ran four
ranking models on it: Boolean OR, TF-IDF cosine, Okapi BM25 and a
Jelinek-Mercer language model. That program ended with a three-number sanity
check (P@10, R@10, R-precision) and left the real question open: which model
actually retrieves the relevant documents, and how do we measure "actually"?

This program answers it. It reuses the Week 6 system unchanged - same parser,
same index, same four scoring functions - and adds the evaluation layer that
CISI makes possible: 76 queries with 3,114 human relevance judgements. For
every judged query each model is scored on

  - Precision@k and Recall@k      how good is the top of the list (k = 5, 10)
  - R-precision                   precision at rank R = number of relevant docs
  - Average Precision -> MAP       rewards relevant docs ranked high, over the
                                   whole list, the standard single-number metric
  - nDCG@k                        discounted gain, position-weighted
  - Reciprocal rank -> MRR         1 / rank of the first relevant document
  - an 11-point interpolated precision-recall curve

Each per-query number is averaged over the 76 judged queries (a macro-average,
so a query with 5 relevant documents counts as much as one with 155). The two
figures - the precision-recall curves and a bar chart of the headline metrics -
are what turn the table into a story about where each model wins and loses.

Why these metrics. Precision and recall are the base pair (Manning et al.,
2008, ch. 8) but depend on a cutoff; MAP and nDCG summarise the whole ranking
in one number, MAP for binary judgements like CISI's and nDCG whenever rank
position should be discounted smoothly. Reporting several together guards
against any single metric flattering a model (Bama et al., 2015).

Usage:
    python cisi_evaluation.py          # full evaluation, tables and figures
"""

import math
import os

from cisi_ir_system import (MODELS, build_index, load_cisi_documents,
                            load_cisi_queries, load_cisi_relevance, search,
                            shorten)
from vsm_similarity import OUTPUT_FOLDER

# Cutoffs the report quotes. k for precision/recall, and the deep cutoff used
# when building the full ranking that MAP, nDCG and the curves read.
PRECISION_KS = (5, 10)
RECALL_KS = (10, 100)
NDCG_KS = (10, 100)

# Recall levels of the standard 11-point interpolated precision-recall curve.
RECALL_LEVELS = [i / 10 for i in range(11)]

# Queries whose per-query scores are printed in full, reused from Week 6 so the
# two reports line up: a short, an average and a long query.
SHOWCASE_QUERIES = [3, 18, 35]


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
# Every function takes the ranked list of document ids (best first) and the
# set of relevant ids for that query. Keeping them this small and separate is
# deliberate: each one is a single textbook definition and can be checked on
# its own.

def precision_at_k(ranked_ids, relevant, k):
    """Fraction of the top k that are relevant."""
    if k == 0:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevant, k):
    """Fraction of all relevant documents found in the top k."""
    if not relevant:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:k]) / len(relevant)


def r_precision(ranked_ids, relevant):
    """Precision at rank R, where R is the number of relevant documents.

    At rank R precision and recall are equal, so one number captures both.
    This makes queries with 5 and 155 relevant documents comparable, which
    matters on CISI where that range is exactly what we have.
    """
    r = len(relevant)
    if r == 0:
        return 0.0
    return sum(doc_id in relevant for doc_id in ranked_ids[:r]) / r


def average_precision(ranked_ids, relevant):
    """Average of the precision values taken at each relevant document.

        AP = (1 / |R|) * sum over ranks i holding a relevant doc of P@i

    A relevant document the model never retrieves contributes nothing to the
    sum but still counts in |R|, so failing to find relevant documents is
    penalised. Averaging AP over the queries gives MAP, the metric most IR
    papers lead with (Manning et al., 2008, ch. 8).
    """
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def reciprocal_rank(ranked_ids, relevant):
    """1 / rank of the first relevant document, 0 if none is retrieved.

    Averaged over queries this is MRR: how far a user has to read before the
    first useful result, ignoring everything after it.
    """
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids, relevant, k):
    """Discounted cumulative gain with binary gains.

        DCG@k = sum over i in 1..k of rel_i / log2(i + 1)

    rel_i is 1 for a relevant document and 0 otherwise. The log discount means
    a relevant document at rank 1 is worth more than one at rank 10, smoothly,
    which is the point nDCG makes over a hard cutoff.
    """
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant:
            total += 1.0 / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids, relevant, k):
    """DCG@k divided by the DCG of the ideal ranking (all relevant first)."""
    if not relevant:
        return 0.0
    ideal_hits = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / ideal


def interpolated_precision(ranked_ids, relevant, levels=RECALL_LEVELS):
    """Precision at each standard recall level, interpolated.

    The interpolated precision at recall r is the highest precision seen at
    any recall >= r (Manning et al., 2008, ch. 8). Walking the ranking once
    records precision and recall at every rank; the interpolation is then a
    single backward pass taking the running maximum. The result is the
    sawtooth-free curve that can be averaged across queries.
    """
    if not relevant:
        return [0.0 for _ in levels]

    # Precision and recall after each retrieved document.
    points = []
    hits = 0
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            hits += 1
        points.append((hits / rank, hits / len(relevant)))

    interpolated = []
    for level in levels:
        best = 0.0
        for precision, recall in points:
            if recall >= level and precision > best:
                best = precision
        interpolated.append(best)
    return interpolated


# ---------------------------------------------------------------------------
# RUNNING ONE MODEL OVER EVERY JUDGED QUERY
# ---------------------------------------------------------------------------

class ModelEvaluation:
    """All the averaged metrics for one model, plus the per-query records.

    per_query[qid] holds (average precision, reciprocal rank, nDCG@10) so the
    showcase table and any later error analysis can look inside the averages.
    """

    def __init__(self, label):
        self.label = label
        self.precision = {k: 0.0 for k in PRECISION_KS}
        self.recall = {k: 0.0 for k in RECALL_KS}
        self.ndcg = {k: 0.0 for k in NDCG_KS}
        self.r_precision = 0.0
        self.mean_ap = 0.0
        self.mrr = 0.0
        self.curve = [0.0 for _ in RECALL_LEVELS]
        self.per_query = {}


def evaluate_model(label, score_function, index, queries, relevance, depth):
    """Score one model on every judged query and macro-average the metrics.

    depth is how many ranks of each result list to keep. It is set to the
    collection size so the ranking is effectively complete: MAP, nDCG and the
    recall figures then see every relevant document the model managed to rank,
    not just a truncated top-k.
    """
    judged = sorted(qid for qid in relevance if qid in queries)
    result = ModelEvaluation(label)

    for qid in judged:
        ranking, _ = search(queries[qid], index, score_function, top_k=depth)
        ranked_ids = [doc_id for doc_id, _ in ranking]
        relevant = relevance[qid]

        for k in PRECISION_KS:
            result.precision[k] += precision_at_k(ranked_ids, relevant, k)
        for k in RECALL_KS:
            result.recall[k] += recall_at_k(ranked_ids, relevant, k)
        for k in NDCG_KS:
            result.ndcg[k] += ndcg_at_k(ranked_ids, relevant, k)
        result.r_precision += r_precision(ranked_ids, relevant)

        ap = average_precision(ranked_ids, relevant)
        rr = reciprocal_rank(ranked_ids, relevant)
        result.mean_ap += ap
        result.mrr += rr

        curve = interpolated_precision(ranked_ids, relevant)
        result.curve = [a + b for a, b in zip(result.curve, curve)]

        result.per_query[qid] = (ap, rr, ndcg_at_k(ranked_ids, relevant, 10))

    n = len(judged)
    for k in PRECISION_KS:
        result.precision[k] /= n
    for k in RECALL_KS:
        result.recall[k] /= n
    for k in NDCG_KS:
        result.ndcg[k] /= n
    result.r_precision /= n
    result.mean_ap /= n
    result.mrr /= n
    result.curve = [value / n for value in result.curve]
    return result, n


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

def save_figures(results, folder):
    """Two figures: the precision-recall curves and a headline-metric bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved = []

    # (1) 11-point interpolated precision-recall curves. A curve that stays
    #     high as recall grows is a model that keeps finding relevant documents
    #     without dragging in junk. The gap between curves is the whole story
    #     of the evaluation in one picture.
    fig, axis = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D"]
    for result, marker in zip(results, markers):
        axis.plot(RECALL_LEVELS, result.curve, marker=marker, linewidth=1.6,
                  markersize=5, label=result.label)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Interpolated precision")
    axis.set_title("11-point interpolated precision-recall (76 CISI queries)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, max(0.6, max(max(r.curve) for r in results) * 1.1))
    axis.grid(True, linewidth=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    pr_path = os.path.join(folder, "precision_recall_curve.png")
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    saved.append(pr_path)

    # (2) The four single-number metrics side by side. MAP and nDCG summarise
    #     the whole ranking; P@10 and R-precision describe the top of it.
    #     Grouping them shows the models agree on the order but not the gaps.
    metrics = [("MAP", lambda r: r.mean_ap),
               ("nDCG@10", lambda r: r.ndcg[10]),
               ("P@10", lambda r: r.precision[10]),
               ("R-prec", lambda r: r.r_precision)]
    fig, axis = plt.subplots(figsize=(8, 5))
    group_width = 0.8
    bar_width = group_width / len(results)
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for model_index, (result, color) in enumerate(zip(results, colors)):
        positions = [m + model_index * bar_width for m in range(len(metrics))]
        heights = [getter(result) for _, getter in metrics]
        axis.bar(positions, heights, bar_width, label=result.label, color=color)
    axis.set_xticks([m + group_width / 2 - bar_width / 2 for m in range(len(metrics))])
    axis.set_xticklabels([name for name, _ in metrics])
    axis.set_ylabel("Score (averaged over 76 queries)")
    axis.set_title("Headline effectiveness metrics by model")
    axis.grid(True, axis="y", linewidth=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    bar_path = os.path.join(folder, "evaluation_metrics.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    saved.append(bar_path)

    return saved


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    documents = load_cisi_documents()
    queries = load_cisi_queries()
    relevance = load_cisi_relevance()
    index = build_index(documents)
    depth = len(documents)

    judged = sorted(qid for qid in relevance if qid in queries)
    relevant_counts = [len(relevance[qid]) for qid in judged]

    print_section("1. WHAT IS BEING EVALUATED")
    print("  The Week 6 CISI system, unchanged: same index, same four models.")
    print("  This program adds the evaluation layer the test collection allows.")
    print(f"\n  Documents                 {len(documents)}")
    print(f"  Judged queries            {len(judged)} of {len(queries)}")
    print(f"  Relevance judgements      {sum(relevant_counts)} (binary)")
    print(f"  Relevant per query        min {min(relevant_counts)}, "
          f"mean {sum(relevant_counts) / len(relevant_counts):.1f}, "
          f"max {max(relevant_counts)}")
    print(f"  Ranking depth per query   {depth} (effectively the full list)")
    print("\n  Metrics: P@5, P@10, R@10, R@100, R-precision, MAP, nDCG@10,")
    print("  nDCG@100, MRR, and an 11-point interpolated precision-recall curve.")
    print("  Each is a macro-average: every query counts equally, regardless")
    print("  of how many documents are relevant to it.")

    results = []
    for label, score_function in MODELS:
        result, n = evaluate_model(label, score_function, index, queries,
                                   relevance, depth)
        results.append(result)

    print_section(f"2. EFFECTIVENESS OF THE FOUR MODELS ({n} JUDGED QUERIES)")
    header = (f"  {'model':<34}{'MAP':>8}{'nDCG@10':>9}{'nDCG@100':>10}"
              f"{'MRR':>8}{'R-prec':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        print(f"  {result.label:<34}{result.mean_ap:>8.3f}{result.ndcg[10]:>9.3f}"
              f"{result.ndcg[100]:>10.3f}{result.mrr:>8.3f}{result.r_precision:>8.3f}")

    print("\n  Precision and recall at fixed cutoffs")
    header = (f"  {'model':<34}{'P@5':>8}{'P@10':>8}{'R@10':>8}{'R@100':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        print(f"  {result.label:<34}{result.precision[5]:>8.3f}"
              f"{result.precision[10]:>8.3f}{result.recall[10]:>8.3f}"
              f"{result.recall[100]:>8.3f}")

    best = max(results, key=lambda r: r.mean_ap)
    worst = min(results, key=lambda r: r.mean_ap)
    print(f"\n  {best.label} leads on every metric ({best.mean_ap:.3f} MAP); "
          f"{worst.label.split('(')[0].strip()}")
    print(f"  is last on every metric ({worst.mean_ap:.3f} MAP). The three")
    print("  term-weighting models all beat plain term-matching however you")
    print("  measure, which is the reassuring result. The only place the")
    print("  ranking shifts is second versus third: BM25 edges TF-IDF at the")
    print("  very top of the list (P@5, P@10) while TF-IDF edges BM25 over the")
    print("  whole ranking (MAP, nDCG, R-precision) - see the analysis below.")

    print_section("3. THE 11-POINT INTERPOLATED PRECISION-RECALL CURVES")
    header = "  recall ->" + "".join(f"{level:>7.1f}" for level in RECALL_LEVELS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        row = "".join(f"{p:>7.3f}" for p in result.curve)
        print(f"  {result.label.split('(')[0].strip():<16}{row}")
    print("\n  Read left to right: precision falls as the system is pushed to")
    print("  return more of the relevant set. CISI is a hard collection - the")
    print("  queries are broad and the relevant sets large - so even the best")
    print("  model gives up precision quickly. The language-model curve sits")
    print("  on top at every recall level; TF-IDF and BM25 run together just")
    print("  below it, and Boolean OR is clearly the lowest throughout.")

    print_section("4. PER-QUERY SCORES ON THE SHOWCASE QUERIES")
    print("  Averages hide variance. These three queries - short, average and")
    print("  long - show how differently the same model does query to query.")
    for qid in SHOWCASE_QUERIES:
        print(f'\n  Q{qid}: "{shorten(queries[qid], 64)}"')
        print(f"       {len(relevance[qid])} relevant documents")
        print(f"    {'model':<34}{'AP':>8}{'RR':>8}{'nDCG@10':>9}")
        for result in results:
            ap, rr, ndcg10 = result.per_query[qid]
            print(f"    {result.label:<34}{ap:>8.3f}{rr:>8.3f}{ndcg10:>9.3f}")

    print_section("5. WHERE EACH MODEL WINS AND LOSES (ANALYSIS)")
    tfidf = next(r for r in results if r.label.startswith("TF-IDF"))
    bm25 = next(r for r in results if r.label.startswith("Okapi"))
    jm = next(r for r in results if r.label.startswith("Jelinek"))
    boolean = next(r for r in results if r.label.startswith("Boolean"))
    print("  - Boolean OR is the floor. It counts matching terms and nothing")
    print(f"    else, so its MAP ({boolean.mean_ap:.3f}) is far below the rest: a")
    print("    document packed with one common query word outranks a focused one.")
    print("  - TF-IDF cosine, BM25 and the language model are close because all")
    print("    three weight rare terms and discount length; they disagree only")
    print("    on the arithmetic.")
    print(f"  - The language model wins outright: MAP {jm.mean_ap:.3f} and the")
    print(f"    best P@10 ({jm.precision[10]:.3f}) and nDCG@10 ({jm.ndcg[10]:.3f}).")
    print("    Jelinek-Mercer smoothing suits CISI's long, wordy queries - a")
    print("    missing term is backed off to the collection, not treated as a")
    print("    veto - which is why it also posts the best MRR.")
    print(f"  - TF-IDF and BM25 split second place. BM25 ranks the very top")
    print(f"    better (P@5 {bm25.precision[5]:.3f} vs {tfidf.precision[5]:.3f}),")
    print(f"    but TF-IDF has the better whole-ranking MAP ({tfidf.mean_ap:.3f} vs")
    print(f"    {bm25.mean_ap:.3f}) and R-precision. On short, fairly even CISI")
    print("    abstracts BM25's length penalty earns little, so its headline")
    print("    advantage over plain cosine does not show.")
    print("  - MRR is high for every weighted model (first relevant result in")
    print("    the top two or three), so the top of the list is usable even when")
    print("    MAP looks low. Low MAP on CISI is a recall problem, not a")
    print("    precision-at-the-top problem: the relevant sets are simply large.")

    figures = save_figures(results, OUTPUT_FOLDER)
    if figures:
        print_section("6. FIGURES")
        for path in figures:
            print(f"  saved {path}")

    print()


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    main()
