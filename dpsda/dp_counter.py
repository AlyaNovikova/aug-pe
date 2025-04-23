# import faiss
import logging
import numpy as np
from collections import Counter
import torch

from scipy.spatial.distance import cdist

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize



def dp_nn_histogram(public_features, private_features, noise_multiplier,
                    num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0):
    assert public_features.shape[0] % num_packing == 0

    num_true_public_features = public_features.shape[0] // num_packing
    if public_features.shape[0] == 0:
        return np.zeros(shape=num_true_public_features), np.zeros(shape=num_true_public_features)

    # Handle different distance metrics
    if mode == 'L2':
        metric = 'euclidean'
    elif mode == 'IP':
        metric = lambda u, v: -np.dot(u, v)  # Using negative since we'll find min instead of max
    elif mode == 'cos_sim':
        # Normalize vectors for cosine similarity
        public_features = public_features / np.linalg.norm(public_features, axis=1, keepdims=True)
        private_features = private_features / np.linalg.norm(private_features, axis=1, keepdims=True)
        metric = lambda u, v: -np.dot(u, v)  # Negative dot product for cosine similarity
    else:
        raise Exception(f'Unknown mode {mode}')

    # Calculate distances between all pairs
    distances = cdist(private_features, public_features, metric=metric)
    
    # Find nearest neighbors
    if num_nearest_neighbor == 1:
        ids = np.argmin(distances, axis=1)
    else:
        # For k>1, get top k indices
        ids = np.argpartition(distances, num_nearest_neighbor, axis=1)[:, :num_nearest_neighbor]
        # Flatten the array to count all nearest neighbors
        ids = ids.flatten()

    counter = Counter(list(ids.flatten()))
    # shape of the synthetic samples
    count = np.zeros(shape=num_true_public_features)
    for k in counter:
        count[k % num_true_public_features] += counter[k]
        
    count = np.asarray(count)
    clean_count = count.copy()
    count += (np.random.normal(size=len(count)) * np.sqrt(num_nearest_neighbor)
              * noise_multiplier)
    count = np.clip(count, a_min=threshold, a_max=None)
    count = count - threshold
    
    return count, clean_count