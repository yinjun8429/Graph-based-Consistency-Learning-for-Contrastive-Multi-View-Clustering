from sklearn.metrics import normalized_mutual_info_score, f1_score, adjusted_rand_score, cluster, accuracy_score, \
    precision_score, recall_score
import sklearn.metrics as metrics
from munkres import Munkres
from sklearn.cluster import KMeans
import numpy as np
import torch
# import hnswlib
import math
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

def normalize1(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x

def get_k_nearest_neighbors(feature_matrix, k=10):




    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')

    nbrs.fit(feature_matrix)
    distances, indices = nbrs.kneighbors(feature_matrix)


    return indices

# def fit_hnsw_index(features, ef=100, M=16, save_index_file=False):
#     # Convenience function to create HNSW graph
#     # features : list of lists containing the embeddings
#     # ef, M: parameters to tune the HNSW algorithm

#     num_elements = len(features)
#     labels_index = np.arange(num_elements)
#     EMBEDDING_SIZE = len(features[0])

#     # Declaring index
#     # possible space options are l2, cosine or ip
#     p = hnswlib.Index(space='l2', dim=EMBEDDING_SIZE)

#     # Initing index - the maximum number of elements should be known
#     p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

#     # Element insertion
#     int_labels = p.add_items(features, labels_index)

#     # Controlling the recall by setting ef
#     # ef should always be > k
#     p.set_ef(ef)

#     # If you want to save the graph to a file
#     if save_index_file:
#         p.save_index(save_index_file)

    return p


def indices2feature(feature, adj_indices, device):
    # feature -- size:(N * D)
    # adj_indices -- size:(N * knn)
    # reture feature_matrix -- size:(N * knn * D)
    feature_matrix = np.zeros((feature.shape[0], adj_indices.shape[1], feature.shape[1]))
    for i in range(feature.shape[0]):
        all = feature.data.cpu().numpy()[adj_indices[i]]
        feature_matrix[i] = all
    feature_matrix = torch.Tensor(feature_matrix).to(device)
    return feature_matrix

def q_distribution_tool(z,n_clusters):
    kmeans = KMeans(n_clusters, n_init=10).fit(z.detach().cpu().numpy())
    cluster_centers = torch.Tensor(kmeans.cluster_centers_).to(z.device)
    norm_squared = torch.sum((z.unsqueeze(1) - cluster_centers) ** 2, 2)
    numerator = 1.0 / (1.0 + (norm_squared / 1.0))
    power = float(1.0 + 1) / 2
    numerator = numerator ** power
    q = (numerator.t() / torch.sum(numerator, 1)).t() # soft assignment using t-distribution
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()