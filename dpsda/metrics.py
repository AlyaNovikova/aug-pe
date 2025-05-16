

"""k-NN precision and recall."""

import numpy as np
from time import time
# example of calculating the frechet inception distance

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


# calculate inception score with Keras
from sklearn.metrics import pairwise_distances

from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *

from bert_score import score

# from metric import calculate_all_metrics_dict, calculate_text_metrics_dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from collections import Counter
import numpy as np
import itertools
import nltk
from scipy.stats import entropy, wasserstein_distance

import pandas as pd

nltk.download('punkt')
nltk.download('punkt_tab')

def get_lengths(texts):
    return [len(nltk.word_tokenize(t)) for t in texts]

def plot_length_distributions(real_lengths, synth_lengths, filename="length_distribution.png"):
    plt.figure(figsize=(10, 6))
    bins = range(0, max(max(real_lengths), max(synth_lengths)) + 5, 1)
    plt.hist(real_lengths, bins=bins, alpha=0.6, label="Real", color='blue', density=True)
    plt.hist(synth_lengths, bins=bins, alpha=0.6, label="Synthetic", color='orange', density=True)
    plt.xlabel("Token Length")
    plt.ylabel("Density")
    plt.title("Distribution of Token Lengths")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def compare_length_distributions(real_lengths, synth_lengths):
    max_len = max(max(real_lengths), max(synth_lengths))
    bins = np.arange(0, max_len + 2) - 0.5  # for integer binning

    real_hist, _ = np.histogram(real_lengths, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_lengths, bins=bins, density=True)

    # Avoid zero probabilities in KL (add small epsilon)
    epsilon = 1e-8
    real_hist += epsilon
    synth_hist += epsilon

    kl_div = entropy(real_hist, synth_hist)
    w_dist = wasserstein_distance(real_lengths, synth_lengths)

    return {
        "length_real_mean": np.mean(real_lengths),
        "length_synthetic_mean": np.mean(synth_lengths),
        "length_real_std": np.std(real_lengths),
        "length_synthetic_std": np.std(synth_lengths),
        "length_kl_divergence": kl_div,
        "length_wasserstein_distance": w_dist
    }

def compute_bleu(real_texts, synthetic_texts):
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(real_texts, synthetic_texts):
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    return np.mean(scores)

def compute_bertscore(real_texts, synthetic_texts, lang='en'):
    P, R, F1 = bert_score(synthetic_texts, real_texts, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

from tqdm import tqdm

def compute_bertscore_pairwise(real_texts, synthetic_texts, lang='en'):
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for syn in tqdm(synthetic_texts, desc="Computing BERTScore pairwise"):
        # Compare current synthetic sample against all real texts
        P, R, F1 = bert_score([syn] * len(real_texts), real_texts, lang=lang, verbose=False)
        all_precisions.append(P.mean().item())
        all_recalls.append(R.mean().item())
        all_f1s.append(F1.mean().item())

    return {
        "pairwise_precision_mean": np.mean(all_precisions),
        "pairwise_precision_std": np.std(all_precisions),
        "pairwise_recall_mean": np.mean(all_recalls),
        "pairwise_recall_std": np.std(all_recalls),
        "pairwise_f1_mean": np.mean(all_f1s),
        "pairwise_f1_std": np.std(all_f1s),
    }

def compute_distinct_2(texts):
    all_bigrams = list(itertools.chain.from_iterable(
        zip(tokens, tokens[1:]) for tokens in [nltk.word_tokenize(t) for t in texts]
    ))
    total_bigrams = len(all_bigrams)
    unique_bigrams = len(set(all_bigrams))
    return unique_bigrams / total_bigrams if total_bigrams > 0 else 0

def compute_self_bleu(texts):
    smoothie = SmoothingFunction().method4
    scores = []
    for i, hyp in enumerate(texts):
        references = texts[:i] + texts[i+1:]
        references_tokenized = [nltk.word_tokenize(ref) for ref in references]
        hyp_tokenized = nltk.word_tokenize(hyp)
        score = sentence_bleu(references_tokenized, hyp_tokenized, smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)



def num_tokens_from_string(string, encoding):
    """Returns the number of tokens in a text string."""
    try:
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens

def calculate_all_metrics_dict(synthetic_embeddings, original_embeddings, k=3):
    import torch
    from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

    # Compute MAUVE and distribution histograms
    p_feats = synthetic_embeddings  # feature dimension = 1024
    q_feats = original_embeddings
    result = compute_mauve(p_feats, q_feats)
    p_hist, q_hist = result.p_hist, result.q_hist
    kl, tv, wass = calculate_other_metrics(p_hist, q_hist)

    # Compute k-NN precision/recall/F1
    state = knn_precision_recall_features(original_embeddings, synthetic_embeddings, nhood_sizes=[k])

    # Compute Sinkhorn loss (Wasserstein-like distance)
    p_feats_torch = torch.from_numpy(synthetic_embeddings)
    q_feats_torch = torch.from_numpy(original_embeddings)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    sinkhorn_loss = loss(p_feats_torch, q_feats_torch).item()

    # Return all metrics in a dictionary
    return {
        "precision": state["precision"],
        "recall": state["recall"],
        "f1": state["f1"],
        "mauve": result.mauve,
        "kl_divergence": kl,
        "total_variation": tv,
        "wasserstein": wass,
        "sinkhorn_loss": sinkhorn_loss,
    }


def calculate_text_metrics_dict(real_text_list, synthetic_data):
    real_trimmed = real_text_list[:len(synthetic_data)]

   
    bert_score = compute_bertscore(real_trimmed, synthetic_data)["f1"]
    
    # bert_score = compute_bertscore_pairwise(real_text_list, synthetic_data)["pairwise_f1_mean"]

    bleu = compute_bleu(real_trimmed, synthetic_data)
    self_bleu = compute_self_bleu(synthetic_data)
    distinct_2 = compute_distinct_2(synthetic_data)

    real_lengths = get_lengths(real_text_list)
    synth_lengths = get_lengths(synthetic_data)

    return {
        "bert_score_f1": bert_score,
        "bleu": bleu,
        "self_bleu": self_bleu,
        "distinct_2": distinct_2,
        "avg_real_length": sum(real_lengths) / len(real_lengths) if real_lengths else 0,
        "avg_synth_length": sum(synth_lengths) / len(synth_lengths) if synth_lengths else 0,
    }

def compute_bertscore(real_texts, synthetic_texts, lang='en'):
    P, R, F1 = bert_score(synthetic_texts, real_texts, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }


from sklearn.metrics.pairwise import cosine_similarity

def average_cosine_similarity(emb_a, emb_b):
    sim_matrix = cosine_similarity(emb_a, emb_b)
    
    return {
        "avg_cosine_sim": sim_matrix.mean(),
        "avg_cosine_sim_a_to_b": sim_matrix.mean(axis=1).mean(), 
        "avg_cosine_sim_b_to_a": sim_matrix.mean(axis=0).mean(),  # Avg for each B vs all A
    }

def vocabulary_overlap(texts_a, texts_b):
    words_a = set(" ".join(texts_a).split())
    words_b = set(" ".join(texts_b).split())
    
    intersection = words_a & words_b
    union = words_a | words_b
    
    return {
        "jaccard_similarity": len(intersection) / len(union) if union else 0,
        "vocab_a_size": len(words_a),
        "vocab_b_size": len(words_b),
    }

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

def word_distribution_similarity(texts_a, texts_b):
    vectorizer = TfidfVectorizer().fit(texts_a + texts_b)
    
    # Convert sparse matrices to dense 1D arrays (average TF-IDF per word)
    tfidf_a = np.asarray(vectorizer.transform(texts_a).mean(axis=0)).flatten()
    tfidf_b = np.asarray(vectorizer.transform(texts_b).mean(axis=0)).flatten()
    
    # Compute cosine similarity (1 - cosine distance)
    cos_sim = 1 - distance.cosine(tfidf_a, tfidf_b)
    return {
        "tfidf_cosine_sim": cos_sim,
        "tfidf_cosine_distance": 1 - cos_sim,  # For consistency
    }

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

def compute_mmd(embeddings_a, embeddings_b):
    K_aa = polynomial_kernel(embeddings_a, embeddings_a)
    K_bb = polynomial_kernel(embeddings_b, embeddings_b)
    K_ab = polynomial_kernel(embeddings_a, embeddings_b)
    
    mmd = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()
    return {"mmd": mmd}

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_self_bleu(texts):
    scores = []
    smoothie = SmoothingFunction().method4
    for i in range(len(texts)):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        scores.append(sentence_bleu(refs, texts[i], smoothing_function=smoothie))
    return {"self_bleu": sum(scores) / len(scores)}

from collections import Counter

def compute_diversity(texts, n_gram=2):
    n_grams = []
    for text in texts:
        tokens = text.split()
        n_grams.extend(zip(*[tokens[i:] for i in range(n_gram)]))
    
    unique_ngrams = set(n_grams)
    diversity = len(unique_ngrams) / len(n_grams) if n_grams else 0
    return {"diversity": diversity}

# def evaluate_text_metrics(synthetic_texts, real_texts):
#     metrics = {}
    
#     # Semantic Quality
#     metrics.update(compute_bertscore(synthetic_texts, real_texts))
    
#     # Diversity
#     metrics.update(compute_diversity(synthetic_texts, n_gram=2))
#     metrics.update(compute_self_bleu(synthetic_texts))
    
#     # Distributional Similarity (Optional)
#     metrics.update(compute_mmd(real_texts, synthetic_texts))
    
#     return metrics

def compare_text_sets(synthetic_texts, real_texts, emb_synth, emb_real):
    metrics = {}
    
    # 1. Semantic similarity
    metrics.update(average_cosine_similarity(emb_synth, emb_real))
    
    # 2. Vocabulary overlap
    metrics.update(vocabulary_overlap(synthetic_texts, real_texts))
    
    # 3. Word frequency (TF-IDF)
    metrics.update(word_distribution_similarity(synthetic_texts, real_texts))
    
    # 4. MMD
    metrics.update(compute_mmd(emb_synth, emb_real))

    metrics.update(calculate_text_metrics_dict(real_texts, synthetic_texts))
    metrics.update(calculate_all_metrics_dict(emb_synth, emb_real))
    
    return metrics

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # Normalize embeddings first
    act1 = (act1 - act1.mean()) / act1.std()
    act2 = (act2 - act2.mean()) / act2.std()
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# ----------------------------------------------------------------------------
# https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py
class DistanceBlock():
    """Provides multi-GPU support to calculate pairwise distances between two batches of feature vectors."""

    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

    def pairwise_distances(self, U, V):
        """Evaluate pairwise distances between two batches of feature vectors."""
        output = pairwise_distances(U, V, n_jobs=24)
        return output


# ----------------------------------------------------------------------------

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, distance_block, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.

            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros(
            [row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch,
                                                                                                       col_batch)

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(
                distance_batch[0:end1 - begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros(
            [num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images, ], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images, ], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch,
                                                                                                       ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1 -
                                                 begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(
                samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps),
                                                    axis=1)
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0:end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


# ----------------------------------------------------------------------------

def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000, num_gpus=1, debug=True):
    """Calculates k-NN precision and recall for two sets of feature vectors.

        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()
    if debug:
        state['precision'] = 0
        state['recall'] = 0
        state['f1'] = 0
        return state

    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold = ManifoldEstimator(
        distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(
        distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0).item()

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0).item()

    state['f1'] = 2 * (state['precision'] * state['recall']) / \
        (state['precision']+state['recall'])

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state


def knn_precision_recall_embeddings(ref_features, eval_features, k=3):
    """
    Calculates k-NN precision, recall, and F1 between two feature sets using cosine distance.

    Args:
        ref_features (np.ndarray): Reference (real) text embeddings.
        eval_features (np.ndarray): Generated (synthetic) text embeddings.
        k (int): Number of neighbors.

    Returns:
        dict: Precision, recall, and F1 score.
    """
    state = {}

    # Normalize features to unit length (cosine distance)
    ref_features = ref_features / np.linalg.norm(ref_features, axis=1, keepdims=True)
    eval_features = eval_features / np.linalg.norm(eval_features, axis=1, keepdims=True)

    # Fit kNN on reference features
    knn_ref = NearestNeighbors(n_neighbors=k, metric='cosine').fit(ref_features)
    distances_eval_to_ref, _ = knn_ref.kneighbors(eval_features)

    precision = np.mean(distances_eval_to_ref[:, -1] < 0.5)  # 0.5 is a common cosine threshold

    knn_eval = NearestNeighbors(n_neighbors=k, metric='cosine').fit(eval_features)
    distances_ref_to_eval, _ = knn_eval.kneighbors(ref_features)

    recall = np.mean(distances_ref_to_eval[:, -1] < 0.5)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    state.update({
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    return state
