import glob
import os

import Stemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def parse_corpus():
    titles = []
    documents = []
    naturalLabels = []
    for naturalLabel in ["athletics", "cricket", "football", "rugby", "tennis"]:
        for file in glob.glob(os.path.join("bbcsport", naturalLabel, "*")):
            currFile = open(file, encoding="utf8", errors="ignore")

            # store each document and title as a string for input
            # to the inverted index
            documents.append(currFile.read())

            # store the natural label of each document
            naturalLabels.append(naturalLabel)

            # go back to the top of a file to obtain the document title
            currFile.seek(0)
            titles.append(currFile.readline().strip())
    return (documents, naturalLabels, titles)


def pre_process_corpus(documents):
    stemmer = Stemmer.Stemmer("english")
    myStopwords = set(stopwords.words("english"))
    myTokenizer = CountVectorizer().build_tokenizer()

    corpusPreProcessed = []
    for document in documents:
        corpusPreProcessed.append(
            " ".join(
                [
                    stemmer.stemWord(i.lower())
                    for i in myTokenizer(document)
                    if i.lower() not in myStopwords
                ]
            )
        )
    return corpusPreProcessed
