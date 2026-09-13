# Information Retrieval System

A basic Information Retrieval (IR) system written in Python 3 for a university assignment.

## Project description

This project demonstrates the core building blocks of an Information Retrieval system on a small
collection of 7 text documents. The program loads the documents, preprocesses the text, builds a
dictionary of unique terms and an inverted index, and then answers Boolean queries (AND, OR, NOT)
using Python set operations.

It is an educational project, not a commercial search engine. Everything is kept deliberately
simple so that each concept (document, term, token, dictionary, inverted index, Boolean retrieval)
can be read directly in the code.

## Assignment objectives

1. Write a program that adds documents to the system.
2. Build a basic dictionary of terms.
3. Form an inverted index for the document collection.
4. Add Boolean retrieval functions (AND, OR, NOT).
5. Produce a report explaining the design and implementation choices.
6. Publish the code on GitHub.

## Features

- Loads all `.txt` files from the `documents/` folder and assigns each one a document ID.
- Simple text preprocessing: lowercase, punctuation removal, splitting into tokens.
- Automatic dictionary generation (sorted list of unique terms) with a term count.
- Inverted index in the form `term -> set of document IDs`.
- Boolean retrieval:
  - `AND` using set intersection (`set1 & set2`)
  - `OR` using set union (`set1 | set2`)
  - `NOT` using set difference (`all_documents - set1`)
- Command-line menu with 7 options.
- Built-in test mode that runs six example queries and compares them with the expected results.
- Queries are case-insensitive, and unknown terms return an empty result instead of crashing.

## Technologies used

- Python 3 (tested on Python 3.12)
- Standard library only: `os`, `string`, `sys`
- No external libraries, no databases, no machine learning, no web framework

## Project structure

```
information-retrieval-system/
│
├── documents/
│   ├── doc1.txt
│   ├── doc2.txt
│   ├── doc3.txt
│   ├── doc4.txt
│   ├── doc5.txt
│   ├── doc6.txt
│   └── doc7.txt
│
├── ir_system.py        # the whole program
├── README.md
├── REPORT_GUIDE.md     # guide for writing the university report
├── requirements.txt
└── .gitignore
```

## How the system works

The program runs in five stages.

**1. Document loading — `load_documents()`**
Every `.txt` file in `documents/` is read and stored in a dictionary:
`{1: {"filename": "doc1.txt", "text": "Artificial intelligence ..."}, ...}`.
Filenames are sorted first, so `doc1.txt` always becomes document ID 1.

**2. Text preprocessing — `preprocess_text()`**
Each document text goes through three steps:

| Step | Purpose | Example |
|------|---------|---------|
| Lowercase | "Machine" and "machine" become the same term | `Machine learning` → `machine learning` |
| Remove punctuation | punctuation is replaced with a space | `statistics,` → `statistics` |
| Split on whitespace | produces a list of tokens | `"machine learning"` → `["machine", "learning"]` |

**3. Dictionary — `build_dictionary()`**
Every token from every document is added to a Python `set`, which automatically removes
duplicates. The set is then sorted into an alphabetical list of unique terms.

**4. Inverted index — `build_inverted_index()`**
For each document, every token is added to a dictionary that maps a term to the **set** of document
IDs containing it. Sets are used because Boolean retrieval is done with set operations, and because
a set stores each document ID only once.

**5. Boolean retrieval**
The three query functions are one line each, because the inverted index has already done the work:

| Query | Function | Set operation |
|-------|----------|---------------|
| `term1 AND term2` | `boolean_and()` | intersection `set1 & set2` |
| `term1 OR term2` | `boolean_or()` | union `set1 \| set2` |
| `NOT term` | `boolean_not()` | difference `all_documents - set1` |

## How to run the program

You need Python 3 installed. No installation of extra packages is required.

```bash
cd information-retrieval-system
python ir_system.py
```

Then choose an option from the menu:

```
====================================
INFORMATION RETRIEVAL SYSTEM
====================================
1. Show documents
2. Show dictionary
3. Show inverted index
4. AND search
5. OR search
6. NOT search
7. Exit
```

To run the built-in tests instead of the menu:

```bash
python ir_system.py test
```

Note: the program must be run from inside the `information-retrieval-system` folder, because it
looks for the `documents/` folder relative to the current directory.

## Example Boolean queries

| Query | Result (document IDs) |
|-------|----------------------|
| `machine AND learning` | `{2, 3, 6}` |
| `artificial AND intelligence` | `{1, 2, 7}` |
| `information AND retrieval` | `{4, 5}` |
| `machine OR data` | `{2, 3, 6, 7}` |
| `artificial OR retrieval` | `{1, 2, 4, 5, 7}` |
| `NOT machine` | `{1, 4, 5, 7}` |

## Example output

Part of the inverted index (option 3):

```
========== INVERTED INDEX ==========
Format:  term -> set of document IDs
Total number of indexed terms: 53

artificial       -> {1, 2, 7}
data             -> {2, 3, 7}
information      -> {4, 5}
intelligence     -> {1, 2, 7}
learning         -> {2, 3, 6}
machine          -> {2, 3, 6}
retrieval        -> {4, 5}
science          -> {3, 7}
```

An AND search (option 4):

```
========== BOOLEAN AND ==========
Enter the first term : machine
Enter the second term: learning

Query: machine AND learning
machine -> {2, 3, 6}
learning -> {2, 3, 6}
Set operation used: intersection  (set1 & set2)
Result (document IDs): {2, 3, 6}
   Doc 2 - Machine learning is a branch of artificial intelligence that learns from data.
   Doc 3 - Data science combines statistics, programming, and machine learning to analyze data.
   Doc 6 - Deep learning is a type of machine learning used for image recognition and natural language processing.
```

A NOT search (option 6):

```
========== BOOLEAN NOT ==========
Enter the term : machine

Query: NOT machine
All documents -> {1, 2, 3, 4, 5, 6, 7}
machine -> {2, 3, 6}
Set operation used: difference  (all_documents - set1)
Result (document IDs): {1, 4, 5, 7}
```

The built-in test mode (`python ir_system.py test`):

```
========== TESTING ==========

Query    : machine AND learning
Expected : {2, 3, 6}
Actual   : {2, 3, 6}
Status   : PASS
...
Tests passed: 6 out of 6
```

## Limitations

- **One operator per query.** Each search compares at most two terms. Combined queries such as
  `machine AND learning OR data`, brackets, and typed query strings are not supported; the terms are
  entered separately through the menu.
- **No stop-word removal.** Common words such as `a`, `is`, `to` and `of` are indexed like any other
  term, which is why the dictionary contains 53 terms.
- **No stemming.** `learning` and `learns` are stored as two different terms, so a search for
  `learning` will not match `learns`.
- **Exact term matching only.** There is no partial matching, spelling correction, or synonym handling.
- **No ranking.** Boolean retrieval returns a set of matching documents, not a ranked list — every
  matching document is considered equally relevant.
- **In-memory only.** The dictionary and inverted index are rebuilt each time the program starts and
  are not saved to disk, which is fine for 7 documents but would not scale.
- **Small fixed collection.** The system reads whatever `.txt` files are in `documents/`, but it has
  only been tested with this 7-document collection.

## Future improvements

- Support multi-term queries with brackets and operator precedence (for example
  `(machine OR data) AND NOT learning`).
- Add a stop-word list to keep the dictionary focused on meaningful terms.
- Add stemming so that `learning` and `learns` are treated as the same term.
- Store term frequencies and add simple ranking such as TF-IDF.
- Save the inverted index to a file so it does not need to be rebuilt on every run.
- Add phrase search using positional information in the index.

## Author

University assignment — basic Information Retrieval system.
