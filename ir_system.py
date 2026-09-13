"""
A basic Information Retrieval (IR) system.

This program:
  1. Loads a small collection of text documents.
  2. Preprocesses the text (lowercase, remove punctuation, split into tokens).
  3. Builds a dictionary of unique terms.
  4. Builds an inverted index (term -> set of document IDs).
  5. Answers Boolean queries (AND, OR, NOT) using Python set operations.

Only the Python standard library is used.
"""

import os
import string
import sys

# Folder that holds the document collection.
DOCUMENTS_FOLDER = "documents"


# ---------------------------------------------------------------------------
# 1. DOCUMENT LOADING
# ---------------------------------------------------------------------------

def load_documents(folder_name):
    """Read every .txt file in the folder and give each one a document ID.

    Returns a dictionary shaped like:
        {1: {"filename": "doc1.txt", "text": "Artificial intelligence ..."}, ...}
    """
    documents = {}

    # Sorting keeps the order predictable, so doc1.txt always becomes ID 1.
    file_names = sorted(f for f in os.listdir(folder_name) if f.endswith(".txt"))

    document_id = 1
    for file_name in file_names:
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "r", encoding="utf-8") as opened_file:
            text = opened_file.read().strip()

        documents[document_id] = {"filename": file_name, "text": text}
        document_id += 1

    return documents


# ---------------------------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_text(text):
    """Turn a piece of raw text into a simple list of tokens (words).

    Three steps only, so the process stays easy to explain:
      1. lowercase the text   -> "Machine" and "machine" become the same term
      2. remove punctuation   -> "data." becomes "data"
      3. split on whitespace  -> one list entry per word
    """
    text = text.lower()

    # string.punctuation holds all the common punctuation characters.
    # Each one is replaced with a space so that words stay separated.
    for punctuation_mark in string.punctuation:
        text = text.replace(punctuation_mark, " ")

    tokens = text.split()
    return tokens


# ---------------------------------------------------------------------------
# 3. DICTIONARY
# ---------------------------------------------------------------------------

def build_dictionary(documents):
    """Collect every unique term that appears anywhere in the collection.

    Returns a sorted list of terms (the "dictionary" or vocabulary).
    """
    unique_terms = set()

    for document_id in documents:
        tokens = preprocess_text(documents[document_id]["text"])
        for token in tokens:
            unique_terms.add(token)

    return sorted(unique_terms)


# ---------------------------------------------------------------------------
# 4. INVERTED INDEX
# ---------------------------------------------------------------------------

def build_inverted_index(documents):
    """Build the inverted index:  term -> set of document IDs containing it.

    A set is used because Boolean retrieval is done with set operations
    (intersection, union and difference).
    """
    inverted_index = {}

    for document_id in documents:
        tokens = preprocess_text(documents[document_id]["text"])
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(document_id)

    return inverted_index


def get_documents_for_term(inverted_index, term):
    """Look a term up in the inverted index.

    Returns the set of document IDs, or an empty set if the term is unknown.
    """
    term = term.lower().strip()
    return inverted_index.get(term, set())


# ---------------------------------------------------------------------------
# 5. BOOLEAN RETRIEVAL
# ---------------------------------------------------------------------------

def boolean_and(inverted_index, first_term, second_term):
    """term1 AND term2  ->  set INTERSECTION  (documents containing both)."""
    first_set = get_documents_for_term(inverted_index, first_term)
    second_set = get_documents_for_term(inverted_index, second_term)
    return first_set & second_set


def boolean_or(inverted_index, first_term, second_term):
    """term1 OR term2  ->  set UNION  (documents containing either term)."""
    first_set = get_documents_for_term(inverted_index, first_term)
    second_set = get_documents_for_term(inverted_index, second_term)
    return first_set | second_set


def boolean_not(inverted_index, all_document_ids, term):
    """NOT term  ->  set DIFFERENCE  (every document that does not contain it)."""
    term_set = get_documents_for_term(inverted_index, term)
    return all_document_ids - term_set


# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------

def format_document_ids(document_ids):
    """Show a set of IDs in a readable, always-sorted way, e.g. {1, 2, 7}."""
    if not document_ids:
        return "{}  (no matching documents)"
    return "{" + ", ".join(str(document_id) for document_id in sorted(document_ids)) + "}"


def show_documents(documents):
    """Print the document collection."""
    print()
    print("========== DOCUMENT COLLECTION ==========")
    print("Number of documents loaded:", len(documents))
    print()
    for document_id in sorted(documents):
        print("Doc ID :", document_id)
        print("File   :", documents[document_id]["filename"])
        print("Text   :", documents[document_id]["text"])
        print("-" * 40)


def show_dictionary(dictionary):
    """Print the dictionary of unique terms."""
    print()
    print("========== DICTIONARY ==========")
    print("Total number of unique terms:", len(dictionary))
    print()

    # Print the terms in rows of 5 so the list fits on the screen.
    terms_per_row = 5
    for position in range(0, len(dictionary), terms_per_row):
        row = dictionary[position:position + terms_per_row]
        print("  ".join(term.ljust(14) for term in row))


def show_inverted_index(inverted_index):
    """Print the inverted index, one term per line."""
    print()
    print("========== INVERTED INDEX ==========")
    print("Format:  term -> set of document IDs")
    print("Total number of indexed terms:", len(inverted_index))
    print()
    for term in sorted(inverted_index):
        print(term.ljust(16), "->", format_document_ids(inverted_index[term]))


def show_result(documents, document_ids):
    """Print the documents that a query matched."""
    print("Result (document IDs):", format_document_ids(document_ids))
    for document_id in sorted(document_ids):
        print("   Doc", document_id, "-", documents[document_id]["text"])
    print()


# ---------------------------------------------------------------------------
# 7. TESTING
# ---------------------------------------------------------------------------

def run_tests(documents, inverted_index, all_document_ids):
    """Run the example queries from the assignment and check the results.

    Each test stores the expected document IDs, so a wrong result is obvious.
    """
    print()
    print("========== TESTING ==========")

    # Each entry: (description, result produced by the system, expected result)
    tests = [
        ("machine AND learning",
         boolean_and(inverted_index, "machine", "learning"), {2, 3, 6}),
        ("artificial AND intelligence",
         boolean_and(inverted_index, "artificial", "intelligence"), {1, 2, 7}),
        ("information AND retrieval",
         boolean_and(inverted_index, "information", "retrieval"), {4, 5}),
        ("machine OR data",
         boolean_or(inverted_index, "machine", "data"), {2, 3, 6, 7}),
        ("artificial OR retrieval",
         boolean_or(inverted_index, "artificial", "retrieval"), {1, 2, 4, 5, 7}),
        ("NOT machine",
         boolean_not(inverted_index, all_document_ids, "machine"), {1, 4, 5, 7}),
    ]

    passed_tests = 0
    for description, actual_result, expected_result in tests:
        if actual_result == expected_result:
            status = "PASS"
            passed_tests += 1
        else:
            status = "FAIL"

        print()
        print("Query    :", description)
        print("Expected :", format_document_ids(expected_result))
        print("Actual   :", format_document_ids(actual_result))
        print("Status   :", status)

    print()
    print("Tests passed:", passed_tests, "out of", len(tests))
    print("=" * 40)


# ---------------------------------------------------------------------------
# 6. USER INTERFACE (command-line menu)
# ---------------------------------------------------------------------------

def print_menu():
    print()
    print("====================================")
    print("INFORMATION RETRIEVAL SYSTEM")
    print("====================================")
    print("1. Show documents")
    print("2. Show dictionary")
    print("3. Show inverted index")
    print("4. AND search")
    print("5. OR search")
    print("6. NOT search")
    print("7. Exit")


def main():
    # ---- Build the system once, at start-up -------------------------------
    if not os.path.isdir(DOCUMENTS_FOLDER):
        print("Error: the folder '" + DOCUMENTS_FOLDER + "' was not found.")
        print("Run this program from inside the project folder.")
        return

    documents = load_documents(DOCUMENTS_FOLDER)
    if not documents:
        print("Error: no .txt files were found in the '" + DOCUMENTS_FOLDER + "' folder.")
        return

    dictionary = build_dictionary(documents)
    inverted_index = build_inverted_index(documents)
    all_document_ids = set(documents.keys())

    print("Loaded", len(documents), "documents from the '" + DOCUMENTS_FOLDER + "' folder.")
    print("Dictionary contains", len(dictionary), "unique terms.")
    print("Inverted index built for", len(inverted_index), "terms.")

    # "python ir_system.py test" runs the built-in tests instead of the menu.
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        run_tests(documents, inverted_index, all_document_ids)
        return

    print("Tip: run 'python ir_system.py test' to run the built-in test queries.")

    # ---- Menu loop --------------------------------------------------------
    while True:
        print_menu()
        try:
            choice = input("Enter your choice (1-7): ").strip()
        except EOFError:
            # Happens when input is piped in and runs out; treat it as Exit.
            print()
            break

        if choice == "1":
            show_documents(documents)

        elif choice == "2":
            show_dictionary(dictionary)

        elif choice == "3":
            show_inverted_index(inverted_index)

        elif choice == "4":
            print()
            print("========== BOOLEAN AND ==========")
            first_term = input("Enter the first term : ").strip()
            second_term = input("Enter the second term: ").strip()
            print()
            print("Query:", first_term, "AND", second_term)
            print(first_term, "->", format_document_ids(get_documents_for_term(inverted_index, first_term)))
            print(second_term, "->", format_document_ids(get_documents_for_term(inverted_index, second_term)))
            print("Set operation used: intersection  (set1 & set2)")
            show_result(documents, boolean_and(inverted_index, first_term, second_term))

        elif choice == "5":
            print()
            print("========== BOOLEAN OR ==========")
            first_term = input("Enter the first term : ").strip()
            second_term = input("Enter the second term: ").strip()
            print()
            print("Query:", first_term, "OR", second_term)
            print(first_term, "->", format_document_ids(get_documents_for_term(inverted_index, first_term)))
            print(second_term, "->", format_document_ids(get_documents_for_term(inverted_index, second_term)))
            print("Set operation used: union  (set1 | set2)")
            show_result(documents, boolean_or(inverted_index, first_term, second_term))

        elif choice == "6":
            print()
            print("========== BOOLEAN NOT ==========")
            term = input("Enter the term : ").strip()
            print()
            print("Query: NOT", term)
            print("All documents ->", format_document_ids(all_document_ids))
            print(term, "->", format_document_ids(get_documents_for_term(inverted_index, term)))
            print("Set operation used: difference  (all_documents - set1)")
            show_result(documents, boolean_not(inverted_index, all_document_ids, term))

        elif choice == "7":
            print()
            print("Exiting the Information Retrieval System. Goodbye!")
            break

        else:
            print()
            print("Invalid choice. Please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()
