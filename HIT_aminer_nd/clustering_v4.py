import codecs
import pandas as pd
import numpy as np
import json
import random
import pickle
import mpi4py.MPI as MPI
import os

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

distance_estimator = pickle.load(open("distance_model", "rb"))


def _affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start + step)
        Xt = np.empty((end - start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]
    return distances


def dbscan_ad():
    with codecs.open("data/sna_data/sna_valid_pub.json", "r", "utf-8") as f:
        pub_valid_dict = json.load(f)

    with codecs.open("data/sna_data/sna_valid_author_raw.json", "r", "utf-8") as f:
        author_raw = json.load(f)
    clusters = {}
    count = 0
    for k, v in author_raw.items():
        print(str(count)+"------"+k)
        count += 1
        n_samples = len(v)
        if not n_samples:
            clusters[k] = []
            continue
        X_ = np.empty((n_samples, 1), dtype=np.object)
        n = 0
        for s in v:
            X_[n, 0] = pub_valid_dict[s]
            n += 1
        #         x_affinity = _affinity(X_)
        #         X = np.zeros((n_samples, n_samples))
        #         X[np.triu_indices(n_samples, 1)] = x_affinity
        #         X = X.transpose()+X
        X = _affinity(X_)
        X = squareform(X)
        length = len(X)
        index = -np.ones((length,), dtype=np.int32)
        labels = np.zeros((length,), dtype=np.int32)
        for cur, i in enumerate(range(length)):
            # if index[cur] != -1:
            #     continue
            big = X[i] > 0.95
            a = np.nonzero(big)[0].tolist()
            a.append(cur)
            cur = labels[cur] if index[cur] != -1 else cur
            index[np.array(a)] = 1
            labels[np.array(a)] = int(cur)
        print(labels)
        paper_dict = {}
        for label, paper in zip(labels, v):
            if label not in paper_dict:
                paper_dict[label] = [paper]
            else:
                paper_dict[label].append(paper)
        clusters[k] = list(paper_dict.values())

    # return clusters
    with codecs.open("cluster_output.json", "w", "utf-8") as wf:
        wf.write(json.dumps(clusters))


def clustering_mpi():
    start = MPI.Wtime()
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    with codecs.open("data/sna_data/sna_valid_pub.json", "r", "utf-8") as f:
        pub_valid_dict = json.load(f)

    with codecs.open("data/rank/" + str(comm_rank) + "/sna_valid_author_raw.json", "r", "utf-8") as f:
        author_raw = json.load(f)
    clusters = {}
    count = 0
    print(comm_rank)
    for k, v in author_raw.items():
        count += 1
        n_samples = len(v)
        if not n_samples:
            clusters[k] = []
            continue
        X_ = np.empty((n_samples, 1), dtype=np.object)
        n = 0
        for s in v:
            X_[n, 0] = pub_valid_dict[s]
            n += 1
        X = _affinity(X_)
        X = squareform(X)
        length = len(X)
        index = -np.ones((length,), dtype=np.int32)
        labels = np.zeros((length,), dtype=np.int32)
        for cur, i in enumerate(range(length)):
            # if index[cur] != -1:
            #     continue
            big = X[i] > 0.95
            a = np.nonzero(big)[0].tolist()
            a.append(cur)
            cur = labels[cur] if index[cur] != -1 else cur
            index[np.array(a)] = 1
            labels[np.array(a)] = int(cur)
        print(str(count) + "------" + k)
        print(labels)
        paper_dict = {}
        for label, paper in zip(labels, v):
            if label not in paper_dict:
                paper_dict[label] = [paper]
            else:
                paper_dict[label].append(paper)
        clusters[k] = list(paper_dict.values())

    if not os.path.exists("data/res"):
        os.makedirs("data/res")
    with codecs.open("data/res/cluster_output" + str(comm_rank) + ".json", "w", "utf-8") as wf:
        wf.write(json.dumps(clusters))
    print("Time used:")
    print(MPI.Wtime() - start)


if __name__ == "__main__":
    dbscan_ad()
    # clustering_mpi()
