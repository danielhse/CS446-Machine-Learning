import torch
import numpy as np
import matplotlib.pyplot as plt


def x_dist(X, centroids):
    dists = np.zeros((X.shape[0], 2)) 
    for i in range(2):
      dists[:, i] = (np.linalg.norm(X - centroids[i, :], axis=1))**2
    return dists

def new_centroids(X, closest):
    centroids = np.zeros((2, X.shape[1]))
    for i in range(2):
      centroids[i, :] = np.mean(X[closest == i, :], axis=0)
    return centroids


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = load_data()

    init_c = init_c.numpy().T 

    centroids = init_c
    cost = 0
    updates = 0

    X = X.numpy().T
    for i in range(n_iters):
      dists = x_dist(X, centroids)
      closest = np.argmin(dists, axis = 1)
      cluster_1 = X[closest == 0, :]
      cluster_2 = X[closest == 1, :]
      
      prev = centroids
      centroids = new_centroids(X, closest)
      cost = tot_cost(X, closest, centroids)
      updates = updates + 1

      if np.linalg.norm(centroids - prev) < 1e-3:
        break

    return torch.Tensor(centroids.T)

def tot_cost(X, closest, centroids):
    dists = np.zeros(X.shape[0])
    for i in range(2):
      dists[closest == i] = np.linalg.norm(X[closest == i] - centroids[i], axis=1)**2
    return np.sum(dists)