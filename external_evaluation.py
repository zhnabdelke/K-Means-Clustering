def frequency(labelsInCluster):

    return {i: labelsInCluster.count(i) for i in labelsInCluster}


def calculate_purity(predictedLabels, naturalLabels):

    purity = 0

    clusterIDs = set(predictedLabels)
    majorityLabelFSum = 0

    for clusterID in clusterIDs:

        # for each distinct predicted cluster ID, retrieve the document indices that
        # correspond to that predicted label (which documents belong to which predicted label)
        documentIndices = [i for i, j in enumerate(predictedLabels) if j == clusterID]

        # for each document index, retrieve the actual cluster ID for that document to
        # calculate the extent to which a cluster contains a single label
        labelsInCluster = [naturalLabels[i] for i in documentIndices]

        # calculate the occurrences of the most frequent label in a cluster
        majorityLabelFSum += max(frequency(labelsInCluster).values())

    purity = majorityLabelFSum / len(naturalLabels)
    return purity
