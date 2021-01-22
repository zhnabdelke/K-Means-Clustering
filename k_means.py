import math

import numpy as np
from numpy import dot
from numpy.linalg import norm

# set seed to get reproduce results
np.random.seed(50)


def cosine_distance(vector1, vector2):

    # cosine distance is defined as 1 - (cosine_similarity(vector1, vector2))
    return 1 - (dot(vector1, vector2) / (norm(vector1) * norm(vector2)))


class KMeans:
    def __init__(self, k, iterations):

        # k is the number of clusters
        self.k = k

        # how many iterations should clusters and centroids be recomputed
        # iterations is one of the termination methods
        self.iterations = iterations

        # initialize each cluster as an empty list
        self.clusters = [[] for i in range(self.k)]

        # initialize the centroids as an empty list
        self.centroids = []

    def make_clusters(self, centroids):

        # initialize the clusters to a list of lists of length k
        clusters = [[] for i in range(self.k)]

        for documentIndex, row in enumerate(self.data):

            # find the index of the centroid such that the cosine distance between
            # a row and a centroid is minimized
            centroidIndex = self.nearest_centroid(row, centroids)

            # append the document index to the cluster that corresponds to the centroid index
            clusters[centroidIndex].append(documentIndex)

        return clusters

    def retrieve_centroids(self, clusters):

        # initialize centroids as nD array of size k * number of index terms
        centroids = np.zeros((self.k, self.numberOfIndexTerms))

        for clusterIndex, documentIndices in enumerate(clusters):

            # the cluster mean is the mean of the data (rows from the TF-IDF matrix)
            # in that cluster
            clusterMean = np.mean(self.data[documentIndices], axis=0)

            # assign the cluster mean to the corresponding centroid
            centroids[clusterIndex] = clusterMean

        return centroids

    def assign_documents_to_cluster(self, data):

        # data is the TF-IDF matrix (documents x index terms)
        self.data = data

        # number of documents is the number of rows in the TF-IDF matrix
        self.numberOfDocuments = data.shape[0]

        # number of index terms is the number of columns in the TF-IDF matrix
        self.numberOfIndexTerms = data.shape[1]

        # retrieve the document indices that will serve as the seed centroids randomly
        seedDocumentIndices = np.random.choice(
            self.numberOfDocuments, self.k, replace=False
        )  # replace = False to avoid selecting a document index more than once

        # assign the centroids to the actual row from the TF-IDF matrix that corresponds to the
        # randomly selected document indices
        self.centroids = [self.data[i] for i in seedDocumentIndices]

        # update clusters and centroids
        for i in range(self.iterations):

            # update clusters
            self.clusters = self.make_clusters(self.centroids)

            # upade the centroids since the clusters were updated in the preceding step
            prevCentroids = self.centroids

            # for each cluster, assign the mean value of the cluster to the corresponding
            # centroid
            self.centroids = self.retrieve_centroids(self.clusters)

            # check for convergence and exit if this occurs
            if self.converges(prevCentroids, self.centroids):
                break

        RSS, topDocuments = self.calculate_RSS(self.clusters, self.centroids)

        # return the cluster label for every document in the TF-IDF matrix
        return (self.cluster_labels(self.clusters), RSS, self.clusters, topDocuments)

    def nearest_centroid(self, row, centroids):

        # compute cosine distances between a document and the centroids
        cosine_distances = [cosine_distance(row, centroid) for centroid in centroids]

        # retrieve the centroid index with the minimum distance with respect
        # to a row from the TF-IDF matrix
        nearestCentroidIndex = np.argmin(cosine_distances)

        return nearestCentroidIndex

    def cluster_labels(self, clusters):

        # initialize the labels as an empty 1D array with a length corresponding
        # to the number of documents in the TF-IDF matrix
        labels = np.empty(self.numberOfDocuments)
        for clusterIndex, cluster in enumerate(clusters):
            for documentIndex in cluster:

                # the label of a document is assigned the index of its cluster,
                # which ranges from 0 to k - 1
                labels[documentIndex] = clusterIndex
        return labels

    def converges(self, prevCentroids, centroids):

        # for each cluster, calculate the cosine distance between the
        # current centroid and the previous centroid
        cosine_distances = [
            cosine_distance(prevCentroids[i], centroids[i]) for i in range(self.k)
        ]

        # convergance occurs when there is no change in the cosine_distance between
        # the previous centroid and the current centroid
        return sum(cosine_distances) == 0

    def calculate_RSS(self, clusters, centroids):

        # store RSS of each cluster
        RSSList = []

        topDocuments = [[] for y in range(self.k)]

        distanceSquared = 0

        for clusterIndex, documentIndices in enumerate(clusters):

            documentsDistanceSquared = []

            for documentIndex in documentIndices:

                # compute the distance squared of a document with respect to its centroid
                distanceSquared = (
                    cosine_distance(self.data[documentIndex], centroids[clusterIndex])
                    ** 2
                )

                documentsDistanceSquared.append(distanceSquared)

                # store the documents that have a distance less than
                # 0.8 with respect to the document's cluster centroid
                distance = math.sqrt(distanceSquared)
                if distance < 0.8:
                    topDocuments[clusterIndex].append((documentIndex, distance))

            # RSS of a cluster is the sum of squared distances between document
            # vectors and their respective centroid
            RSSList.append(sum(documentsDistanceSquared))

        return (RSSList, topDocuments)
