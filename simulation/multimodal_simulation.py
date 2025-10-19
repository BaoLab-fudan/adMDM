from __future__ import print_function
import numpy as np
import random
from copy import deepcopy

# for the replication of the same simulation data, uncomment next line
np.random.seed(2022)
import pandas as pd

""" Function to generate simulation data with two modalities """


def basic_simulator(n_clusters,
                    n,
                    d_1,
                    d_2,
                    k,
                    sigma_1,
                    sigma_2,
                    decay_coef_1,
                    decay_coef_2
                    ):
    """
    Generate simulated data with two modalities.
    Parameters:
      n_clusters:       number of ground truth clusters.
      n:                number of cells to simulate.
      d_1:              dimension of features for modality X.
      d_2:              dimension of features for modality Y.
      k:                dimension of latent code to generate simulate data (for both modality)
      sigma_1:          variance of gaussian noise for modality X.
      sigma_2:          variance of gaussian noise for modality Y.
      decay_coef_1:     decay coefficient of dropout rate for modality X.
      decay_coef_2:     decay coefficient of dropout rate for modality Y.

    Output:
      a dataframe with keys as follows
      'true_cluster':   true cell clusters, a vector of length n
      'data_a_full':    feature matrix of X without dropouts
      'data_a_dropout': feature matrix of X with dropouts
      'data_a_label':   cluster labels to generate X
      'data_b_full':    feature matrix of Y without dropouts
      'data_b_dropout': feature matrix of Y with dropouts
      'data_b_label':   cluster labels to generate Y
    Altschuler & Wu Lab 20212
    Software provided as is under MIT License.
    """

    # data dict for output
    data = {}

    """ generation of true cluster labels """
    cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
    data['true_cluster'] = cluster_ids

    """ split a group of true clusters randomly """
    # divide clusters into two equal groups
    section_a = np.arange(np.floor(n_clusters / 2.0))
    section_b = np.arange(np.floor(n_clusters / 2.0), n_clusters)

    uniform_a = np.random.uniform(size=section_a.size)
    uniform_b = np.random.uniform(size=section_b.size)

    cluster_ids_a = cluster_ids.copy().astype(float)
    cluster_ids_b = cluster_ids.copy().astype(float)

    # reindex
    cluster_ids_a = pd.Series(cluster_ids_a)
    cluster_ids_a = cluster_ids_a.map(dict(zip(cluster_ids_a.unique(), range(cluster_ids_a.unique().size))))
    cluster_ids_a = cluster_ids_a.values

    cluster_ids_b = pd.Series(cluster_ids_b)
    cluster_ids_b = cluster_ids_b.map(dict(zip(cluster_ids_b.unique(), range(cluster_ids_b.unique().size))))
    cluster_ids_b = cluster_ids_b.values

    """ Simulation of modality A"""
    # generate latent code
    Z_a = np.zeros([k, n])
    for id in list(set(cluster_ids_a)):
        idxs = cluster_ids_a == id
        cluster_mu = (np.random.random([k]) - .5)
        Z_a[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()

    # first layer of neural network
    A_a_1 = np.random.random([d_1, k]) - 0.5
    X_a_1 = np.dot(A_a_1, Z_a)
    X_a_1 = 1 / (1 + np.exp(-X_a_1))

    # second layer of neural network
    A_a_2 = np.random.random([d_1, d_1]) - 0.5
    noise_a = np.random.normal(0, sigma_1, size=[d_1, n])
    X_a = (np.dot(A_a_2, X_a_1) + noise_a).transpose()
    X_a = 1 / (1 + np.exp(-X_a))

    # random dropouts
    Y_a = deepcopy(X_a)
    rand_matrix = np.random.random(Y_a.shape)
    zero_mask = rand_matrix < decay_coef_1
    Y_a[zero_mask] = 0

    data['data_a_full'] = X_a
    data['data_a_dropout'] = Y_a
    data['data_a_label'] = cluster_ids_a

    """ Simulation of modality A"""
    # generate latent code
    Z_b = np.zeros([k, n])
    for id in list(set(cluster_ids_b)):
        idxs = cluster_ids_b == id
        cluster_mu = (np.random.random([k]) - .5)
        Z_b[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()

    # first layer of neural network
    A_b_1 = np.random.random([d_2, k]) - 0.5
    X_b_1 = np.dot(A_b_1, Z_b)
    X_b_1 = 1 / (1 + np.exp(-X_b_1))

    # second layer of neural network
    A_b_2 = np.random.random([d_2, d_2]) - 0.5
    noise_b = np.random.normal(0, sigma_2, size=[d_2, n])
    X_b = (np.dot(A_b_2, X_b_1) + noise_b).transpose()
    X_b = 1 / (1 + np.exp(-X_b))

    # random dropouts
    Y_b = deepcopy(X_b)
    rand_matrix = np.random.random(Y_b.shape)
    zero_mask = rand_matrix < decay_coef_2
    Y_b[zero_mask] = 0

    data['data_b_full'] = X_b
    data['data_b_dropout'] = Y_b
    data['data_b_label'] = cluster_ids_b

    return data
