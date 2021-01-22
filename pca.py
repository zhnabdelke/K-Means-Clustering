import numpy as np
from numpy.linalg import eig


class PCA:
    def __init(self, targetDimensions):

        self.targetDimensions = targetDimensions
        self.pc = None
        self.mean = None

    def retrieve_components(self, data):

        self.mean = np.mean(data, axis=0)

        data -= self.mean

        # transpose the data and create the covariance matrix
        # where the covariance is with respect to the N features
        covariance = np.cov(data.T)

        eigenvalues = eig(covariance)[0]
        eigenvectors = eig(covariance)[1]

        # transpose the eigenvectors (column vectors)
        # to facilitate calculations
        eigenvectors = eigenvectors.T

        # sort the eigenvalues in descending order based on value
        # and store the sorted indices
        eigenvaluesIndices = np.argsort(eigenvalues)[::-1]

        # store the sorted eigenvalues and eigenvectors
        eigenvalues = eigenvalues[eigenvaluesIndices]
        eigenvectors = eigenvectors[eigenvaluesIndices]

        # the components are the eigenvectors from indices 0 to N
        # target dimensions
        self.pc = eigenvectors[: self.targetDimensions]

    def project_data(self, data):
        data -= self.mean
        return np.dot(data, self.pc)
