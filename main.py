import numpy as np
from tabulate import tabulate

from external_evaluation import calculate_purity
from invert import InvertedIndex
from k_means import KMeans
from pre_processing import parse_corpus, pre_process_corpus

corpus, naturalLabels, titles = parse_corpus()
corpusPreProcessed = pre_process_corpus(corpus)

naturalLabels = [naturalLabels.index(l) for l in naturalLabels]


def generate_matrix(corpusPreProcessed):
    invertedIndex = InvertedIndex()

    for i in range(len(corpusPreProcessed)):
        for term in corpusPreProcessed[i].split():
            invertedIndex.parse_term(term, i)
    documentXTermMatrix = np.array(invertedIndex.make_document_by_term_array())
    return documentXTermMatrix


matrix = generate_matrix(corpusPreProcessed)
k = KMeans(5, 200)

documentClusters = k.assign_documents_to_cluster(matrix)

predictedLabels = documentClusters[0]

cluster_tightness = documentClusters[1]

clusters = documentClusters[2]

topDocuments = documentClusters[3]


def write_clusters():
    with open("clusters.txt", "w") as f:
        for i in range(len(clusters)):
            data = []

            f.write(
                "Cluster #%d contains the following %d documents: "
                % (i, len(clusters[i]))
            )
            f.write("\n\n")
            for j in range(len(clusters[i])):
                id = clusters[i][j]
                data.append([id, titles[id]])
            f.write(tabulate(data, headers=["Document ID", "Document Title"]))
            f.write("\n\n")


def sort_tuples(tuples):

    # sort tuples in ascending order by the second element
    # (distance from the centroid), which acts as the key

    tuples.sort(key=lambda x: x[1])
    return tuples


def show_summary():
    for i in range(len(topDocuments)):
        data = []
        print("The top 3 documents in cluster #%d are:\n " % i)
        sortedTuples = sort_tuples(topDocuments[i])[:3]
        for j in sortedTuples:
            data.append([j[0], titles[j[0]]])
        print(tabulate(data, headers=["Document ID", "Document Title"]))
        print()


def show_RSS():
    data = []
    for i in range(len(cluster_tightness)):
        data.append([i, cluster_tightness[i]])

    print(tabulate(data, headers=["Cluster ID", "RSS"]))

    print("\nThe total RSS is %f." % sum(cluster_tightness))


def show_purity():
    purity = calculate_purity(predictedLabels, naturalLabels)
    print("The purity is %f." % purity)


def display_menu():

    # display menu shown to user
    print("")
    print("*" * 60, "Menu", "*" * 60)
    print("1. Show Cluster Summary")
    print("2. Calculate RSS")
    print("3. Calculate Purity")
    print("4. Write Clusters")
    print("5. Exit")
    print("*" * 128)
    print("")


def wait_for_input():
    input("\nPlease press Enter to continue...")


status = True

# main loop to display the menu
while status:
    display_menu()
    selection = input("Please enter your selection (1-4): ")
    print()
    if selection == "1":
        show_summary()
        wait_for_input()

    elif selection == "2":
        show_RSS()
        wait_for_input()

    elif selection == "3":
        show_purity()
        wait_for_input()

    elif selection == "4":
        write_clusters()
        wait_for_input()

    elif selection == "5":
        print("\nThe program will now terminate.")
        status = False

    else:

        # prompt user for a valid selection
        input("Please select a valid option from the menu.\n")
