import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC


def load_data(data_path, vocab_size=14141):
    """Load data from data_path

    Returns:
        data: a matrix, i-th column is a vector representing tf_idf values of i-th document
        labels: a row vector, i-th element is the label of i-th document
    """

    data = []
    labels = []

    with open(data_path) as f:
        lines = f.readlines()

    for line in lines:
        features = line.split("<fff>")
        label, doc_id = int(features[0]), int(features[1])
        sparse_r_d = features[2]
        indices_tfidfs = sparse_r_d.split()
        r_d = [0.0 for _ in range(vocab_size)] # a column of data

        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(":")[0])
            tfidf = float(index_tfidf.split(":")[1])
            r_d[index] = tfidf

        data.append(np.array(r_d))
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    print("-----Load data successfully-----")
    return data, labels


def clustering_with_KMeans(data):
    """KMeans clustering using sklearn."""

    # use csr matrix to create a sparse matrix with efficient row slicing
    X = csr_matrix(data)
    print('=================')
    kmeans = KMeans(
        n_clusters=20, # number of clusters
        init='random', # random initialization
        n_init=5, # number of time that kmeans runs with differently initialized centroids
        tol=1e-3, # threshold for acceptable minimum error decrease
        random_state=2018 # set to get deterministic results
    ).fit(X)
    return kmeans.labels_


def classifying_with_linear_SVMs(train_X, train_y, test_X, test_y):
    """Training data using linear SVM

    Args:
        train_X: a matrix, i-th column is a vector representing tf_idf values of i-th document
        train_Y: a row vector, i-th element is the label of i-th document
        test_X, test_y: data used to test the classifier

    """

    classifier = LinearSVC(
        C=1.0, # penalty coeff
        tol=0.001, # tolerance for stopping criteria
        verbose=True # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)

    # compute Y_predicted
    train_y_predicted = classifier.predict(train_X)
    test_y_predicted = classifier.predict(test_X)

    # compute accuracy
    training_accuracy = compute_accuracy(train_y_predicted, train_y)
    test_accuracy = compute_accuracy(test_y_predicted, test_y)

    # print the training and test accuracy
    print(f"Training accuracy: {training_accuracy}")
    print(f"Test accuracy: {test_accuracy}")


def compute_accuracy(predicted_Y, expected_Y):
    """Compute the accuracy."""
    matches = np.equal(predicted_Y, expected_Y)
    accuracy = np.sum(matches.astype(float)) / expected_Y.size
    return accuracy


if __name__ == '__main__':
    # kmeans
    data, labels = load_data("./datasets/data_tf_idf.txt")
    predicted_labels_kmeans = clustering_with_KMeans(data)

    # linear SVM
    train_X, train_y = load_data("./datasets/20news_train_tfidf.txt")
    test_X, test_y = load_data("./datasets/20news_test_tfidf.txt")
    classifying_with_linear_SVMs(train_X, train_y, test_X, test_y)


