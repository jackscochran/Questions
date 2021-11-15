import nltk
import sys
import os
import string
import re
import math
import random

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    # files = load_files(sys.argv[1])
    files = load_files('corpus')
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        f = open(path, encoding="utf8")
        files[filename] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    words = [
        word.lower() for word in
        nltk.word_tokenize(document)
        if word.isalpha() or is_number(word)
    ]

    return words


def is_number(string):
    if string.isnumeric():
        return True

    try:
        return float(string) and '.' in string  # True if string is a number contains a dot
    except ValueError:  # String is not a number
        return False


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_idfs = dict()

    words = set()

    for document in documents:
        words.update(documents[document])

    for word in words:
        f = sum(word in documents[document] for document in documents)
        word_idfs[word] = math.log(len(documents) / f)


    return word_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    tfidfs = dict()

    for file in files:
        tfidfs[file] = 0
        for keyword in query:
            f = 0
            for word in files[file]:
                if word == keyword:
                    f +=1
            tf = f / len(file)
            tfidfs[file] += tf * idfs[keyword]


    best_tfidfs = []

    for i in range(n):
        best_file = max(tfidfs, key=tfidfs.get)
        best_tfidfs.append(best_file)
        del tfidfs[best_file]

    return best_tfidfs


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = dict()

    for sentence in sentences:
        sentence_scores[sentence] = 0
        for word in sentences[sentence]:
            if word in query:
                sentence_scores[sentence] += idfs[word]


    best_sentences = []

    for i in range(n):
        best_sentence = max(sentence_scores, key=sentence_scores.get)
        best_sentences.append(best_sentence)
        del sentence_scores[best_sentence]


    return best_sentences


if __name__ == "__main__":
    main()
