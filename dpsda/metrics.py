import numpy as np
from time import time

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

from sklearn.metrics import pairwise_distances

from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *

from bert_score import score

from bert_score import score as bert_score
from collections import Counter
import numpy as np
import itertools
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import entropy, wasserstein_distance

import wandb
import pandas as pd

import spacy
import itertools
import numpy as np

nltk.download('punkt', force=True)  

print(nltk.data.path)

# Tokenizer factory function
def get_tokenizer(language='english', backend='nltk'):
    if backend == 'nltk':
        nltk.download('punkt', quiet=True)
        # nltk.download('punkt_tab')

        def tokenizer(text):
            return nltk.word_tokenize(text, language=language)
        return tokenizer

    elif backend == 'spacy':
        model_map = {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'sci': 'en_core_sci_sm',
        }
        if language not in model_map:
            raise ValueError(f"Unsupported language '{language}' for spaCy.")
        nlp = spacy.load(model_map[language])
        def tokenizer(text):
            return [token.text for token in nlp(text)]
        return tokenizer

    else:
        raise ValueError("Unsupported backend. Choose 'nltk' or 'spacy'.")


def get_lengths(texts, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return [len(tokenizer(t)) for t in texts]

def plot_length_distributions(real_lengths, synth_lengths, filename="length_distribution.png"):
    plt.figure(figsize=(10, 6))
    bins = range(0, max(max(real_lengths), max(synth_lengths)) + 5, 1)
    plt.hist(real_lengths, bins=bins, alpha=0.6, label="Real", color='blue',
             density=True, edgecolor='black', rwidth=0.9)
    plt.hist(synth_lengths, bins=bins, alpha=0.6, label="Synthetic", color='orange',
             density=True, edgecolor='black', rwidth=0.9)
    plt.xlabel("Token Length")
    plt.ylabel("Density")
    plt.title("Distribution of Token Lengths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
        "length_mean_diff": abs(np.mean(real_lengths) - np.mean(synth_lengths)),

        "length_real_mean": np.mean(real_lengths),
        "length_synthetic_mean": np.mean(synth_lengths),
        "length_real_std": np.std(real_lengths),
        "length_synthetic_std": np.std(synth_lengths),
        "length_kl_divergence": kl_div,
        "length_wasserstein_distance": w_dist
    }

def compute_bleu(real_texts, synthetic_texts, tokenizer):
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(real_texts, synthetic_texts):
        ref_tokens = tokenizer(ref)
        hyp_tokens = tokenizer(hyp)
        scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    return np.mean(scores)

from tqdm import tqdm


def compute_bertscore(real_texts, synthetic_texts, lang='en'):
    if len(real_texts) != len(synthetic_texts):
        real_text_list = real_texts[:len(synthetic_texts)]
    else:
        real_text_list = real_texts
    P, R, F1 = bert_score(synthetic_texts, real_text_list, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

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

def compute_distinct_2(texts, tokenizer):
    all_bigrams = list(itertools.chain.from_iterable(
        zip(tokens, tokens[1:]) for tokens in [tokenizer(t) for t in texts]
    ))
    total_bigrams = len(all_bigrams)
    unique_bigrams = len(set(all_bigrams))
    return unique_bigrams / total_bigrams if total_bigrams > 0 else 0

def compute_self_bleu(texts, tokenizer):
    smoothie = SmoothingFunction().method4
    scores = []
    for i, hyp in enumerate(texts):
        references = texts[:i] + texts[i+1:]
        references_tokenized = [tokenizer(ref) for ref in references]
        hyp_tokenized = tokenizer(hyp)
        score = sentence_bleu(references_tokenized, hyp_tokenized, smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)

from rouge_score import rouge_scorer

def compute_rouge_l(real_texts, synthetic_texts):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for ref, hyp in zip(real_texts, synthetic_texts):
        score = scorer.score(ref, hyp)['rougeL'].fmeasure
        scores.append(score)

    return np.mean(scores)

def vocabulary_overlap(texts_a, texts_b, tokenizer):
    words_a = set(word.lower() for text in texts_a for word in tokenizer(text))
    words_b = set(word.lower() for text in texts_b for word in tokenizer(text))
    
    intersection = words_a & words_b
    union = words_a | words_b
    
    return {
        "jaccard_similarity": len(intersection) / len(union) if union else 0,
        # "vocab_a_size": len(words_a),
        # "vocab_b_size": len(words_b),
    }


def num_tokens_from_string(string, encoding):
    try:
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens

# def calculate_all_metrics(original_embeddings, synthetic_embeddings, k=3):
#     method_name = ""
#     p_feats = synthetic_embeddings  # feature dimension = 1024
#     q_feats = original_embeddings
#     result = compute_mauve(p_feats, q_feats)
#     print("MAUVE: ", result.mauve)
#     p_hist, q_hist = result.p_hist, result.q_hist
#     kl, tv, wass = calculate_other_metrics(p_hist, q_hist)

#     state = knn_precision_recall_features(
#         original_embeddings, synthetic_embeddings, nhood_sizes=[k])
#     print(state)

#     from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

#     # feature dimension = 1024
#     p_feats = torch.from_numpy(synthetic_embeddings)
#     q_feats = torch.from_numpy(original_embeddings)

#     # Define a Sinkhorn (~Wasserstein) loss between sampled measures
#     loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

#     # By default, use constant weights = 1/number of samples
#     sinkhorn_loss = loss(p_feats, q_feats).item()
#     print("Sinkhorn loss: %.3f" % sinkhorn_loss)

#     return state['precision'], state['recall'], state['f1'], result.mauve, kl, tv, wass, sinkhorn_loss

def calculate_all_metrics_dict(original_embeddings, synthetic_embeddings, k=3, real_texts=None, synthetic_texts=None):
    import torch
    from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    print("original_embeddings and synthetic_embeddings shape", original_embeddings.shape, synthetic_embeddings.shape)

    fid = calculate_fid(original_embeddings, synthetic_embeddings)
    print("METRICS", "fid1", fid)

    ref_features = normalize(original_embeddings, axis=1)
    synt_features = normalize(synthetic_embeddings, axis=1)

    fid = calculate_fid(ref_features, synt_features)
    print("METRICS", "fid2", fid)

    # Compute MAUVE and distribution histograms
    p_feats = synt_features  # feature dimension = 1024
    q_feats = ref_features

    result = compute_mauve_score(real_texts, synthetic_texts)
    print("METRICS", "mauve 1", result.mauve)

    p_hist, q_hist = result.p_hist, result.q_hist
    kl, tv, wass = calculate_other_metrics(p_hist, q_hist)

    # Compute k-NN precision/recall/F1
    # state = knn_precision_recall_features(ref_features, synt_features, nhood_sizes=[5, 10, 15])
    max_len = min(min(len(ref_features), len(synt_features)) - 1, 5)
    print('Max neighbors for precision and recall', max_len)
    state = knn_precision_recall_features(ref_features, synt_features, nhood_sizes=[max_len])
    print("METRICS", "precision_recall1--", state["precision"], state["recall"])

    state = knn_precision_recall_features(original_embeddings, synthetic_embeddings, nhood_sizes=[max_len])
    print("METRICS", "precision_recall3--", state["precision"], state["recall"])

    # Compute Sinkhorn loss (Wasserstein-like distance)
    p_feats_torch = torch.from_numpy(synt_features)
    q_feats_torch = torch.from_numpy(ref_features)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    sinkhorn_loss = loss(p_feats_torch, q_feats_torch).item()

    # Return all metrics in a dictionary
    return {
        "precision": state["precision"],
        "recall": state["recall"],
        "f1": state["f1"],
        "mauve": result.mauve,
        "kl_divergence": kl,
        "FID": fid,
        # "total_variation": tv,
        # "wasserstein": wass,
        "sinkhorn_loss": sinkhorn_loss,
    }


def calculate_text_metrics_dict(real_text_list, synthetic_data, tokenizer):
    real_trimmed = real_text_list[:len(synthetic_data)]
   
    bert_score = compute_bertscore(real_trimmed, synthetic_data)["f1"]
    
    # bert_score = compute_bertscore_pairwise(real_text_list, synthetic_data)["pairwise_f1_mean"]

    bleu = compute_bleu(real_trimmed, synthetic_data, tokenizer)
    self_bleu = compute_self_bleu(synthetic_data, tokenizer)
    distinct_2 = compute_distinct_2(synthetic_data, tokenizer)

    real_lengths = get_lengths(real_text_list, tokenizer)
    synth_lengths = get_lengths(synthetic_data, tokenizer)

    length_dict = compare_length_distributions(real_lengths, synth_lengths)

    rouge = compute_rouge_l(real_text_list, synthetic_data)

    return {
        "bert_score_f1": bert_score,
        "bleu": bleu,
        "self_bleu": self_bleu,
        "distinct_2": distinct_2,
        "length_mean_diff": length_dict["length_mean_diff"],
        "Rouge-L": rouge,
    }



def plot_metrics(metrics_history, output_dir):
    """Plot all metrics across epochs and save figures."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = sorted(metrics_history.keys())
    metrics_names = list(metrics_history[epochs[0]].keys())
    
    # Log each metric with epoch as step
    for epoch in epochs:
        metrics_to_log = {"epoch": epoch}
        for metric in metrics_names:
            metrics_to_log[f"metrics/{metric}"] = metrics_history[epoch][metric]
        wandb.log(metrics_to_log)
    
    # # Plot each metric separately
    # for metric in metrics_names:
    #     plt.figure(figsize=(10, 6))
    #     values = [metrics_history[e][metric] for e in epochs]
    #     plt.plot(epochs, values, marker='o')
    #     plt.title(f'{metric} across epochs')
    #     plt.xlabel('Epoch')
    #     plt.ylabel(metric)
    #     plt.grid(True)
    #     plot_path = os.path.join(output_dir, f'{metric}.png')
    #     plt.savefig(plot_path)
    #     plt.close()
        
    #     # Log to wandb
    #     wandb.log({f"plots/{metric}": wandb.Image(plot_path)})
    
    # # Plot all metrics together (normalized)
    # plt.figure(figsize=(12, 8))
    # for metric in metrics_names:
    #     values = [metrics_history[e][metric] for e in epochs]
    #     # Normalize for visualization
    #     values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
    #     plt.plot(epochs, values, marker='o', label=metric)
    # plt.title('All metrics across epochs (normalized)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Normalized value')
    # plt.legend()
    # plt.grid(True)
    # combined_path = os.path.join(output_dir, 'all_metrics_normalized.png')
    # plt.savefig(combined_path)
    # plt.close()
    
    # # Log to wandb
    # wandb.log({"plots/all_metrics_normalized": wandb.Image(combined_path)})

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(real_features, synthetic_features, method="tsne", title="Embeddings Visualization", n_components=2, filename=""):
    # Stack and create labels
    all_embeddings = np.vstack([real_features, synthetic_features])
    from sklearn.preprocessing import normalize
    import umap
    all_embeddings = normalize(all_embeddings)

    labels = np.array(["Real"] * len(real_features) + ["Synthetic"] * len(synthetic_features))

    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)

    else:
        raise ValueError("Unsupported method: use 'tsne'")

    reduced_embeddings = reducer.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(10, 6))
    for label, color in zip(["Real", "Synthetic"], ["blue", "orange"]):
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=label, alpha=0.7, c=color)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    plt.savefig(filename)
    plt.close()


def compare_text_sets(real_texts, synthetic_texts, emb_real, emb_synth, result_folder="", epoch=0):

    plots_folder = os.path.join(result_folder, "plots_metrics")
    os.makedirs(plots_folder, exist_ok=True)
    plot_embeddings(emb_real, emb_synth, filename=os.path.join(plots_folder, f"embeddings_{epoch}.png"), n_components=3, method="umap")

    metrics = {}

    tokenizer = get_tokenizer()

    metrics.update(vocabulary_overlap(real_texts, synthetic_texts, tokenizer))
    metrics.update(calculate_text_metrics_dict(real_texts, synthetic_texts, tokenizer))
    metrics.update(calculate_all_metrics_dict(emb_real, emb_synth, real_texts=real_texts, synthetic_texts=synthetic_texts))
    
    return metrics

# calculate frechet inception distance
def calculate_fid(act1, act2, regularize=True):
    # Normalize embeddings first
    # act1 = (act1 - act1.mean()) / act1.std()
    # act2 = (act2 - act2.mean()) / act2.std()
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # Regularize covariance matrices
    if regularize:
        epsilon = 1e-6
        sigma1 += np.eye(sigma1.shape[0]) * epsilon
        sigma2 += np.eye(sigma2.shape[0]) * epsilon

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1 @ sigma2)
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

# def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
#                                   row_batch_size=10000, col_batch_size=50000, num_gpus=1, debug=True):
#     """Calculates k-NN precision and recall for two sets of feature vectors.

#         Args:
#             ref_features (np.array/tf.Tensor): Feature vectors of reference images.
#             eval_features (np.array/tf.Tensor): Feature vectors of generated images.
#             nhood_sizes (list): Number of neighbors used to estimate the manifold.
#             row_batch_size (int): Row batch size to compute pairwise distances
#                 (parameter to trade-off between memory usage and performance).
#             col_batch_size (int): Column batch size to compute pairwise distances.
#             num_gpus (int): Number of GPUs used to evaluate precision and recall.

#         Returns:
#             State (dict): Dict that contains precision and recall calculated from
#             ref_features and eval_features.
#     """
#     state = dict()
#     if debug:
#         state['precision'] = 0
#         state['recall'] = 0
#         state['f1'] = 0
#         return state

#     num_images = ref_features.shape[0]
#     num_features = ref_features.shape[1]

#     # Initialize DistanceBlock and ManifoldEstimators.
#     distance_block = DistanceBlock(num_features, num_gpus)
#     ref_manifold = ManifoldEstimator(
#         distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
#     eval_manifold = ManifoldEstimator(
#         distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

#     # Evaluate precision and recall using k-nearest neighbors.
#     print('Evaluating k-NN precision and recall with %i samples...' % num_images)
#     start = time()

#     # Precision: How many points from eval_features are in ref_features manifold.
#     precision = ref_manifold.evaluate(eval_features)
#     state['precision'] = precision.mean(axis=0).item()

#     # Recall: How many points from ref_features are in eval_features manifold.
#     recall = eval_manifold.evaluate(ref_features)
#     state['recall'] = recall.mean(axis=0).item()

#     state['f1'] = 2 * (state['precision'] * state['recall']) / \
#         (state['precision']+state['recall'])

#     print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

#     return state


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
