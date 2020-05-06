from collections import defaultdict

import numpy as np
import random


class Member:
    """
    A class used to represent a single document d

    -----
    Attributes
    -----
    _r_d : list of int
        tf_idf values of the document d
    _label : int
        the id group that d belongs to
    _doc_id : int
        the id of the file that contains d

    """

    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    """
    A class used to represent a cluster

    -----
    Attributes
    -----
    _centroid : int
        the centroid of the cluster
    _members : list
        members of the cluster

    -----
    Methods
    -----
    reset_member: clear all members of the cluster
    add_member(member): add member to the cluster
    """

    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_member(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)


class Kmeans:
    """
    A class that used to implement k-means algorithm

    -----
    Attributes
    -----
    _num_clusters : int
        the number of clusters
    _clusters : list
        a list of Clusters(object)
    _E: list
        a list of centroids
    _S: float
        the overall similarity value of all clusters
    _data: list
        a list of Members(object)
    _label_count: dictionary
        mapping from label to the number of members

    -----
    Methods
    -----
    load_data(data_path): load data from data path

    random_init(seed_value): randomly initialize centroids

    compute_similarity(member, centroid): compute similarity between the member and centroid using rbf kernel
    select_cluster_for(member): select a cluster for the member with largest similarity
    update_centroid_of(cluster): update the centroid of the cluster
    stopping_condition(criterion, threshold): check the stopping condition
    run(seed_value, criterion, threshold): run the algorithm.

    compute_purity, compute_NMI: evaluate the clustering
    """

    # -------------------
    # Initialize clusters
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(num_clusters)]

        self._E = []  # list of centroids
        self._S = 0  # overall similarity
        self._data = []  # list of members

    # ------------------------
    # Load data from data_path
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            #
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()

            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(":")[0])
                tfidf = float(index_tfidf.split(":")[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()

        with open("./datasets/words_idfs.txt") as f:
            vocab_size = len(f.readlines())

        self._label_count = defaultdict(int)
        # key: label
        # value: the number of members with that label

        for d in d_lines:
            features = d.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            sparse_r_d = features[2]
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=sparse_r_d, vocab_size=vocab_size)
            self._data.append(Member(r_d, label, doc_id))

    # -----------------
    # K-means algorithm
    def random_init(self, seed_value):
        """Initialize centroids
        Choose centroids randomly from data points.
        """

        random.seed(seed_value)
        self._E = []
        rand_indices = random.sample(range(len(self._data)), self._num_clusters)  # get random indices of data

        # assign centroids to these data points
        for index, rand_index in enumerate(rand_indices):
            centroid = self._data[rand_index]._r_d
            self._clusters[index]._centroid = centroid
            self._E.append(centroid)

    def compute_similarity(self, member, centroid, gamma=1):
        """Compute similarity between a member's tf_idfs and a centroid using RBF kernel."""

        rbf = np.exp(-gamma * (np.linalg.norm(member._r_d - centroid)))
        return rbf

    def select_cluster_for(self, member):
        """
        Select a cluster which has the largest similarity to the member

        Returns:
            the value of the biggest similarity measure
        """

        best_fit_cluster = None
        max_similarity = -1

        # loop over clusters
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)

            # select the cluster with higher similarity
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity

        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        """Update the centroid of the cluster."""

        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])
        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        """Check the stopping condition
        3 criteria: centroid, similarity, iterations.
        centroid:
            the number new centroids < threshold
        iterations:
            the number of iterations > threshold
        similarity:
            the increase in value of similarity measure < threshold
        """

        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria

        if criterion == 'max_iters':
            if self._iteration >= threshold:
                return True
            else:
                return False

        if criterion == 'similarity':
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

        if criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new__minus_E = [centroid for centroid in E_new
                              if centroid not in self._E]
            self._E = E_new
            if len(E_new__minus_E) <= threshold:
                return True
            else:
                return False

    def run(self, seed_value, criterion, threshold):
        """
        Run the algorithm.
        First initialize the centroids. Then repeat (select cluster for every data points -->
        update centroid of every clusters --> check the stopping condition)
        """

        # initialize centroids randomly
        self.random_init(seed_value)

        # continually update cluster until convergence
        self._iteration = 0
        while True:

            # reset cluster, retain only centroids
            for cluster in self._clusters:
                cluster.reset_member()
            self._new_S = 0

            # select cluster for each data point
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s

            # update the centroid of each cluster
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            # check the stopping condition
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

    # -----------------------
    # Evaluate the clustering
    def compute_purity(self):
        # purity
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        # normalized mutual information
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label
                             for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)


if __name__ == '__main__':

    # Initialize centroids 5 times and choose the one with highest purity or NMI.
    num_clusters = 20
    kmeans = Kmeans(num_clusters)
    kmeans.load_data("./datasets/data_tf_idf.txt")

    max_purity = 0
    best_seed_value = 0
    centroids = []

    for seed_value in range(5):
        kmeans.run(seed_value, "max_iters", 20)
        purity = kmeans.compute_purity()

        # select the cluster with higher purity
        if purity > max_purity:
            max_purity = purity
            best_seed_value = seed_value
            centroids = kmeans._E
