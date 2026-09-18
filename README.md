# Week 3 – Document Similarity with the Vector Space Model

TECH 400: Introduction to Information Retrieval — Week 3 assignment.

The program represents eight Wikipedia articles as TF-IDF vectors and measures
how similar they are with cosine similarity. It also ranks the documents
against free-text queries.

## How it works

1. **Load** every `.txt`, `.docx` or `.pdf` file in `documents/`.
2. **Preprocess**: lowercase → keep alphabetic tokens → remove stop words → Porter stemming.
3. **Weight** each term with `(1 + log10 tf) × log10(N / df)`.
4. **Normalize** each document vector to unit length.
5. **Compare** documents with cosine similarity (dot product of unit vectors).
6. **Rank** documents for a query using the same weighting.
7. **Verify** the nearest neighbours against scikit-learn's `TfidfVectorizer`.

The TF-IDF and cosine code is written by hand. scikit-learn is used only for the check.

## Run

```bash
pip install -r requirements.txt
python vsm_similarity.py                        # full output with sample queries
python vsm_similarity.py "solar panels"         # also rank your own query
```

Output from the last run is in `output/program_output.txt` and the heatmap is in
`output/similarity_heatmap.png`.

## Main result

| Most similar pair | Cosine |
|---|---|
| Climate change ↔ Renewable energy | 0.257 |
| Machine learning ↔ Deep learning | 0.242 |
| Information retrieval ↔ Search engine | 0.119 |

Least similar pair: Vector space model ↔ Association football (0.017).

Document sources and license: see [`documents/SOURCES.md`](documents/SOURCES.md).
