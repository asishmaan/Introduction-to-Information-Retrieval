# Report Guide

This file is a guide for writing your university report on this project. It is **not** the report
itself — it tells you what to write in each section, which part of the code to refer to, and which
screenshot to insert.

Take all screenshots yourself by running the program. Do not use output copied from this guide
without running it first.

---

## Screenshot checklist (take these first)

Run the program from inside the `information-retrieval-system` folder.

| # | Screenshot | How to produce it |
|---|-----------|-------------------|
| 1 | Documents loaded | `python ir_system.py`, then choose **1** |
| 2 | Dictionary | choose **2** |
| 3 | Inverted index | choose **3** |
| 4 | AND search | choose **4**, enter `machine`, then `learning` |
| 5 | OR search | choose **5**, enter `machine`, then `data` |
| 6 | NOT search | choose **6**, enter `machine` |
| 7 | Menu + testing | screenshot the menu itself, and run `python ir_system.py test` |

Tips for clean screenshots:

- Maximise the terminal window first so nothing wraps awkwardly.
- Capture the section heading (`========== ... ==========`) in the shot — it makes the
  screenshot self-explanatory in the report.
- Number every screenshot in your report ("Figure 1: Document collection loaded") and refer to it
  in the text ("as shown in Figure 1").

---

## 1. Introduction

**What to explain**

- What Information Retrieval is: finding documents relevant to a user's query from a collection.
- Why a naive approach does not work: scanning every document for every query is slow, so IR
  systems build an index in advance.
- What this project is: a basic IR system over 7 short text documents, built in Python 3 using only
  the standard library, that demonstrates preprocessing, dictionary construction, inverted index
  construction, and Boolean retrieval.
- State clearly that this is an educational system, not a commercial search engine.

**Code it relates to:** the docstring at the top of `ir_system.py` (lines 1–12).

**Screenshot:** none needed.

---

## 2. Objectives

**What to explain**

List the objectives from the assignment brief, in your own words:

1. Create a small document collection (7 documents).
2. Write a program that loads documents into the system.
3. Preprocess text into tokens.
4. Build a dictionary of unique terms automatically.
5. Build an inverted index mapping terms to document IDs.
6. Implement Boolean retrieval (AND, OR, NOT).
7. Provide a simple user interface.
8. Test the system and report the results.

**Code it relates to:** the whole program; you can mention that each objective maps to one function.

**Screenshot:** none needed.

---

## 3. System Design

**What to explain**

- The overall pipeline, in order:

  ```
  Documents  ->  Preprocessing  ->  Dictionary
                       |
                       v
                 Inverted Index  ->  Boolean Retrieval  ->  Results
  ```

  Draw this as a simple diagram in your report.

- **Design choice: functions, not classes.** The system is small and each stage is a single
  transformation of data, so plain functions keep it readable. A class would add structure without
  adding clarity at this size.
- **Design choice: sets for postings.** The inverted index stores document IDs in a Python `set`
  because (a) a document ID should appear only once per term, and (b) Boolean retrieval is exactly
  set intersection, union and difference, so the operations become one line each.
- **Design choice: sorted dictionary.** The dictionary is returned as a sorted list so the output is
  stable and easy to read in a screenshot.
- **Design choice: build once at start-up.** The index is built when the program starts, and every
  query afterwards is just a set lookup. Explain that this is the whole point of an index.
- **Design choice: standard library only.** No external packages, so the program runs anywhere
  Python 3 is installed.

**Code it relates to:** the section headers in `ir_system.py`, and `main()` (line 268), where the
documents, dictionary and index are built once before the menu loop starts.

**Screenshot:** none needed (use your own diagram instead).

---

## 4. Document Collection

**What to explain**

- The collection contains 7 short `.txt` documents in the `documents/` folder.
- Why the documents were chosen: they are on overlapping topics (artificial intelligence, machine
  learning, data science, information retrieval), so terms are shared between documents. This makes
  AND, OR and NOT queries produce interesting, non-trivial results.
- How document IDs are assigned: filenames are sorted, then numbered from 1. So `doc1.txt` is
  document 1, `doc2.txt` is document 2, and so on. Explain why sorting matters — without it the IDs
  could change between runs and the report screenshots would not match.
- List the 7 documents and their text in a table.

**Code it relates to:** `load_documents()` (line 26), and the constant `DOCUMENTS_FOLDER` (line 19).

**Screenshot 1 — Documents loaded** (menu option 1).

**What it should demonstrate:** that all 7 files were read successfully, that each has a document ID
and a filename, and that the stored text matches the file contents. Point out the line
`Number of documents loaded: 7`.

---

## 5. Text Preprocessing

**What to explain**

- Why preprocessing is needed: raw text contains capital letters and punctuation, so `"Machine"`,
  `"machine"` and `"machine,"` would otherwise be treated as three different terms.
- The three steps, with an example for each:
  1. **Lowercasing** — `"Machine Learning"` → `"machine learning"`.
  2. **Punctuation removal** — each punctuation character is replaced with a space, so
     `"statistics,"` → `"statistics"`. Explain why it is replaced with a *space* rather than deleted:
     it avoids accidentally joining two words together.
  3. **Tokenisation** — `.split()` breaks the text on whitespace into a list of tokens.
- Define the difference between a **token** (one occurrence of a word in a document) and a **term**
  (one distinct word in the vocabulary). Doc 3 contains the token `data` twice, but `data` is a
  single term.
- **Justify keeping it simple.** Say explicitly that stop-word removal and stemming were *not*
  implemented, and why: the assignment asks for simple preprocessing, and leaving them out makes the
  behaviour of the index easy to verify by hand. Mention them again in Limitations.

**Code it relates to:** `preprocess_text()` (line 53).

**Screenshot:** none required, but you can include a short code screenshot of `preprocess_text()`
if your report benefits from showing the code.

---

## 6. Dictionary Construction

**What to explain**

- What the dictionary (vocabulary) is: the set of all unique terms appearing anywhere in the
  collection.
- How it is built: every document is tokenised, every token is added to a Python `set`, and the set
  removes duplicates automatically. The set is then sorted alphabetically into a list.
- Why a `set` was used instead of checking a list with `if token not in list`: sets handle
  uniqueness for you and lookups are fast.
- State that the dictionary is generated **automatically** from the documents — nothing is typed in
  by hand. This is an explicit assignment requirement, so say it clearly.
- Report the size: **53 unique terms** across the 7 documents. Comment on why the number is
  relatively high — no stop words are removed, so `a`, `is`, `of`, `to` are all counted as terms.

**Code it relates to:** `build_dictionary()` (line 76) and `show_dictionary()` (line 170).

**Screenshot 2 — Dictionary** (menu option 2).

**What it should demonstrate:** the alphabetical list of unique terms and the line
`Total number of unique terms: 53`. Point out that terms such as `artificial`, `learning` and
`information` appear exactly once in the list even though they occur in several documents.

---

## 7. Inverted Index Construction

**What to explain**

- **What an inverted index is.** A normal ("forward") index maps a document to the words it
  contains. An inverted index reverses this: it maps a term to the documents that contain it. That
  reversal is what makes retrieval fast — you look up the term once instead of scanning every
  document.
- The structure used: a Python dictionary where the key is a term (string) and the value is a set of
  document IDs (integers): `term -> {doc IDs}`.
- How it is built: loop over each document, tokenise it, and for each token add the current document
  ID to that term's set. If the term is new, create an empty set for it first.
- Why a set and not a list: a term may occur several times in the same document (for example `data`
  in doc 3), and a set stores the document ID only once, automatically. Also, sets support the
  operations Boolean retrieval needs.
- Walk through a few real entries from your own output, for example:

  ```
  artificial   -> {1, 2, 7}
  machine      -> {2, 3, 6}
  information  -> {4, 5}
  ```

  Verify one of them by hand in the report: show that `machine` really does appear in documents
  2, 3 and 6 and nowhere else. This proves the index is correct rather than just plausible.

  Note: the example index in the assignment brief was illustrative. In this actual collection,
  `machine` appears in document 3 as well ("...and machine learning to analyze data"), so the
  correct posting set is `{2, 3, 6}`. Mentioning this shows you verified the output rather than
  copying the brief.

**Code it relates to:** `build_inverted_index()` (line 95), `get_documents_for_term()` (line 113),
and `show_inverted_index()` (line 184).

**Screenshot 3 — Inverted index** (menu option 3).

**What it should demonstrate:** the full `term -> set of document IDs` mapping, the line
`Total number of indexed terms: 53`, and clearly visible entries for `artificial`, `machine`,
`learning`, `data`, `information` and `retrieval`.

---

## 8. Boolean Retrieval

**What to explain first (shared introduction)**

- The Boolean retrieval model: a query is a logical expression over terms, and a document either
  matches or it does not. There is no ranking and no partial match.
- Because postings are stored as sets, each Boolean operator maps directly onto one set operation.
  Make this mapping a table in your report.

| Operator | Meaning | Set operation | Python |
|----------|---------|---------------|--------|
| AND | documents containing **both** terms | intersection | `set1 & set2` |
| OR | documents containing **either** term | union | `set1 \| set2` |
| NOT | documents **not** containing the term | difference | `all_documents - set1` |

- Explain that unknown terms return an **empty set** rather than an error, because
  `get_documents_for_term()` uses `.get(term, set())`. An empty set still works correctly in all
  three operations.
- Explain that query terms are lowercased before lookup, so `Machine` and `machine` give the same
  result. This mirrors the preprocessing applied when the index was built — the query must be
  processed the same way as the documents, otherwise lookups would fail.

### 8.1 AND — set intersection

**What to explain**

- `machine AND learning` returns documents that contain both terms.
- Show the working:
  `machine -> {2, 3, 6}`, `learning -> {2, 3, 6}`, intersection `= {2, 3, 6}`.
- A better example for showing that intersection really filters:
  `artificial AND intelligence` → `{1, 2, 7}`, or try `machine AND data`
  (`{2, 3, 6} & {2, 3, 7} = {2, 3}`) where the result is smaller than either input.
- State the property: an AND result is always a subset of each input set, so it can only ever get
  smaller.

**Code it relates to:** `boolean_and()` (line 126) and menu option 4 in `main()`.

**Screenshot 4 — AND search** (menu option 4, terms `machine` and `learning`).

**What it should demonstrate:** the two posting sets, the line
`Set operation used: intersection  (set1 & set2)`, the resulting document IDs, and the matching
document texts printed underneath so the reader can confirm both terms really appear.

### 8.2 OR — set union

**What to explain**

- `machine OR data` returns documents that contain either term (or both).
- Show the working: `machine -> {2, 3, 6}`, `data -> {2, 3, 7}`, union `= {2, 3, 6, 7}`.
- Note that document 2 and 3 contain both terms but appear only once in the result — the set handles
  the duplicate automatically.
- State the property: an OR result is always at least as large as each input set.

**Code it relates to:** `boolean_or()` (line 133) and menu option 5 in `main()`.

**Screenshot 5 — OR search** (menu option 5, terms `machine` and `data`).

**What it should demonstrate:** the two posting sets, the line `Set operation used: union  (set1 | set2)`,
and the combined result `{2, 3, 6, 7}` with the matching documents listed.

### 8.3 NOT — set difference

**What to explain**

- `NOT machine` returns every document that does **not** contain the term.
- Explain why the full set of document IDs is needed: NOT is only meaningful relative to the whole
  collection. The program keeps `all_document_ids = set(documents.keys())` for exactly this reason.
- Show the working: `all_documents = {1..7}`, `machine -> {2, 3, 6}`, difference `= {1, 4, 5, 7}`.
- Mention that this is a *complement* operation, and that in a very large collection a NOT query
  could return almost the entire collection — which is why real systems usually only allow NOT in
  combination with another operator (for example `A AND NOT B`).

**Code it relates to:** `boolean_not()` (line 140) and menu option 6 in `main()`.

**Screenshot 6 — NOT search** (menu option 6, term `machine`).

**What it should demonstrate:** the line `All documents -> {1, 2, 3, 4, 5, 6, 7}`, the posting set
`machine -> {2, 3, 6}`, the line `Set operation used: difference  (all_documents - set1)`, and the
result `{1, 4, 5, 7}`. Point out that none of the listed documents contains the word "machine".

---

## 9. Implementation

**What to explain**

- Language and environment: Python 3 (state the version you ran it on), standard library only.
- The file structure of the project — reuse the tree from `README.md`.
- A table of the main functions and what each one does:

| Function | Line | Responsibility |
|----------|------|----------------|
| `load_documents()` | 26 | reads the `.txt` files and assigns document IDs |
| `preprocess_text()` | 53 | lowercase, remove punctuation, split into tokens |
| `build_dictionary()` | 76 | collects unique terms into a sorted list |
| `build_inverted_index()` | 95 | builds `term -> set of document IDs` |
| `get_documents_for_term()` | 113 | safe lookup, returns an empty set for unknown terms |
| `boolean_and()` | 126 | set intersection |
| `boolean_or()` | 133 | set union |
| `boolean_not()` | 140 | set difference against all document IDs |
| `run_tests()` | 207 | runs the six example queries and checks them |
| `main()` | 268 | builds the system once, then runs the menu loop |

- Explain the separation of responsibilities: the retrieval functions return sets, and the `show_*`
  functions do the printing. Keeping them separate means the retrieval logic can be tested without
  looking at printed output.
- Explain the user interface: a `while True` loop that prints the menu, reads a choice with
  `input()`, and calls the matching function. Option 7 breaks the loop. Invalid input prints a
  message and shows the menu again instead of crashing.
- Mention the small robustness touches, since they show deliberate design:
  - the program checks that the `documents/` folder exists before starting;
  - unknown query terms return an empty set instead of raising `KeyError`;
  - query terms are lowercased so input capitalisation does not matter.

**Screenshot:** optionally a screenshot of the code itself (for example the three Boolean functions
together, lines 126–144) — they fit on one screen and show all three set operations at once.

---

## 10. Testing and Results

**What to explain**

- How testing was done: the program includes a test mode, run with `python ir_system.py test`. Each
  test stores the **expected** document IDs alongside the query, and the program compares them with
  the actual result and prints PASS or FAIL.
- Why expected results were worked out by hand first: with only 7 short documents you can verify
  every answer manually, which is what makes the test meaningful.
- Present the results table:

| # | Query | Expected | Actual | Status |
|---|-------|----------|--------|--------|
| 1 | `machine AND learning` | `{2, 3, 6}` | `{2, 3, 6}` | PASS |
| 2 | `artificial AND intelligence` | `{1, 2, 7}` | `{1, 2, 7}` | PASS |
| 3 | `information AND retrieval` | `{4, 5}` | `{4, 5}` | PASS |
| 4 | `machine OR data` | `{2, 3, 6, 7}` | `{2, 3, 6, 7}` | PASS |
| 5 | `artificial OR retrieval` | `{1, 2, 4, 5, 7}` | `{1, 2, 4, 5, 7}` | PASS |
| 6 | `NOT machine` | `{1, 4, 5, 7}` | `{1, 4, 5, 7}` | PASS |

  (Confirm these against your own run before submitting — all six passed when the program was last run.)

- Describe at least one manual verification in prose. For example, for query 6: documents 2, 3 and 6
  contain "machine", so the remaining documents 1, 4, 5 and 7 are correctly returned.
- Mention the extra cases you tested beyond the six required queries:
  - a term that is not in the dictionary (e.g. `banana AND machine`) → empty result, no crash;
  - capitalised input (e.g. `Machine` and `Healthcare`) → same behaviour as lowercase;
  - an invalid menu choice (e.g. `9`) → error message, menu redisplayed.

**Code it relates to:** `run_tests()` (line 207).

**Screenshot 7 — Menu and testing.** Take two shots here:
 (a) the menu itself, and (b) the output of `python ir_system.py test`.

**What they should demonstrate:** (a) that the interface offers all seven options as specified;
(b) the Expected / Actual / Status lines for each query and the final line
`Tests passed: 6 out of 6`.

---

## 11. Discussion

**What to explain**

- **Why the inverted index is the right structure.** Compare the two approaches: scanning every
  document for every query costs time proportional to the size of the whole collection, while an
  index lookup goes straight to the answer. Note that the index is built once and reused for every
  query afterwards.
- **Why sets were the right choice.** They remove duplicate document IDs automatically, and they
  turn each Boolean operator into a single built-in operation. Say honestly that this made the
  retrieval functions almost trivially short — the real work happened when the index was built.
- **What the trade-off is.** Building the index costs time and memory up front, and the index must
  be rebuilt if the documents change. For 7 documents this is invisible; for a large collection it
  is the central engineering problem.
- **What this model cannot do.** Boolean retrieval gives a yes/no answer. A user searching for
  `machine AND learning` gets three documents in no particular order, with no indication of which is
  most relevant. Real systems add ranking (e.g. TF-IDF, BM25) for this reason.
- **What you learned.** Write this genuinely — for example, that the choice of data structure
  determined how simple the retrieval code could be, or that preprocessing decisions directly change
  what the system can find (`learns` vs `learning`).

**Screenshot:** none needed.

---

## 12. Limitations

**What to explain**

Take these from `README.md` and explain *why* each one exists, rather than just listing them:

- **One operator per query.** The menu asks for two terms and applies one operator. Parsing a full
  query string with brackets and precedence would need a query parser, which is beyond the scope of
  a basic system.
- **No stop-word removal.** Words like `a`, `is`, `of` are indexed, which is why the dictionary has
  53 terms. They add no retrieval value but were kept so the preprocessing stays simple and
  transparent.
- **No stemming.** `learning` and `learns` are separate terms, so a search for `learning` misses
  document 2's `learns`. Give this concrete example — it shows you understand the consequence.
- **Exact matching only.** No partial matches, spelling correction or synonyms.
- **No ranking.** Results are a set; every matching document is treated as equally relevant.
- **In-memory only.** The index is rebuilt on every run and not saved to disk.
- **Small collection.** Only tested on these 7 documents.

Be honest here. Correctly identifying limitations usually earns more marks than overstating what the
system does.

**Screenshot:** none needed.

---

## 13. Conclusion

**What to explain**

- Restate briefly what was built: a working basic IR system covering document loading,
  preprocessing, dictionary construction, inverted index construction and Boolean retrieval.
- State that all objectives were met and that all six test queries returned the expected document
  IDs.
- Mention one or two concrete next steps (ranking with TF-IDF, multi-operator queries, stop-word
  removal and stemming).
- Close with what the exercise demonstrated about IR: that the index, not the query code, is where
  the work happens.

**Screenshot:** none needed.

---

## 14. GitHub Repository

**What to explain**

- State that the code is published on GitHub and give the repository link.
- Optionally mention what the repository contains (the program, the 7 documents, the README, the
  `.gitignore`).

**Insert your repository URL here when you have created it:**

```
https://github.com/<your-username>/<your-repository-name>
```

Check before submitting that the repository is **public** (or that your instructor has access), and
open the link in a private browser window to confirm it loads.

**Screenshot:** optional — a screenshot of the GitHub repository page showing the files is a nice
final figure.

---

## Final checklist before submitting

- [ ] All 7 screenshots taken from your own run of the program.
- [ ] Every figure numbered and referred to in the text.
- [ ] Document IDs in the report match the IDs in the screenshots.
- [ ] The 53-term dictionary count matches your screenshot.
- [ ] Test results table matches the test output screenshot.
- [ ] GitHub link inserted and tested.
- [ ] Report describes only features the program actually has.
