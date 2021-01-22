import math
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.indexTerms = {}
        self.corpus = Counter()

    def parse_term(self, indexTerm, documentID):

        acc = 1

        if documentID not in self.corpus:
            self.corpus[documentID] = 0

        if indexTerm not in self.indexTerms:
            self.indexTerms[indexTerm] = Counter()

        self.corpus[documentID] += acc
        self.indexTerms[indexTerm][documentID] += acc

    def calculate_term_frequency(self, indexTerm, documentID):
        return self.indexTerms[indexTerm].get(documentID, 0)

    def calculate_document_frequency(self, indexTerm):
        return len(self.indexTerms[indexTerm])

    def calculate_tfidf(self, indexTerm, documentID):
        tfRaw = self.calculate_term_frequency(indexTerm, documentID)
        if not tfRaw == 0:
            tf = 1 + math.log10(tfRaw)
        else:
            tf = 0

        if not tf == 0:
            n = len(self.corpus)
            df = self.calculate_document_frequency(indexTerm)
            idf = math.log10(n / df)
            return tf * idf

        else:
            return 0

    def make_document_vector(self, documentID):
        documentVector = []
        for indexTerm in self.indexTerms:
            documentVector.append(self.calculate_tfidf(indexTerm, documentID))
        return documentVector

    def make_document_by_term_array(self):
        document_by_term_array = []
        for documentID in self.corpus:
            document_by_term_array.append(self.make_document_vector(documentID))
        return document_by_term_array
