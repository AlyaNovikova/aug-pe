import tiktoken
import numpy as np
from time import time
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import os
import matplotlib.pyplot as plt
import wandb

# calculate inception score with Keras
import torch
import argparse
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset

from dpsda.metrics import calculate_fid

from dpsda.logging import *
from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *
from apis.utils import set_seed

import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


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


def calculate_all_metrics(synthetic_embeddings, original_embeddings, k=3):
    method_name = ""
    p_feats = synthetic_embeddings  # feature dimension = 1024
    q_feats = original_embeddings
    result = compute_mauve(p_feats, q_feats)
    print("MAUVE: ", result.mauve)
    p_hist, q_hist = result.p_hist, result.q_hist
    kl, tv, wass = calculate_other_metrics(p_hist, q_hist)

    state = knn_precision_recall_features(
        original_embeddings, synthetic_embeddings, nhood_sizes=[k])
    print(state)

    from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

    # feature dimension = 1024
    p_feats = torch.from_numpy(synthetic_embeddings)
    q_feats = torch.from_numpy(original_embeddings)

    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    # By default, use constant weights = 1/number of samples
    sinkhorn_loss = loss(p_feats, q_feats).item()
    print("Sinkhorn loss: %.3f" % sinkhorn_loss)

    return state['precision'], state['recall'], state['f1'], result.mauve, kl, tv, wass, sinkhorn_loss


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
    
    # Plot each metric separately
    for metric in metrics_names:
        plt.figure(figsize=(10, 6))
        values = [metrics_history[e][metric] for e in epochs]
        plt.plot(epochs, values, marker='o')
        plt.title(f'{metric} across epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb
        wandb.log({f"plots/{metric}": wandb.Image(plot_path)})
    
    # Plot all metrics together (normalized)
    plt.figure(figsize=(12, 8))
    for metric in metrics_names:
        values = [metrics_history[e][metric] for e in epochs]
        # Normalize for visualization
        values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
        plt.plot(epochs, values, marker='o', label=metric)
    plt.title('All metrics across epochs (normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized value')
    plt.legend()
    plt.grid(True)
    combined_path = os.path.join(output_dir, 'all_metrics_normalized.png')
    plt.savefig(combined_path)
    plt.close()
    
    # Log to wandb
    wandb.log({"plots/all_metrics_normalized": wandb.Image(combined_path)})


def eval_one_file(syn_fname, all_original_embeddings, model, csv_fname, batch_size, private_data_size, num_run, k, dataset="yelp", min_token_threshold=100, epoch=None, 
                  real_file="", synthetic_folder=""):
    syn_data = load_dataset("csv", data_files=syn_fname)

    synthetic_data = []
    if dataset == "yelp":
        for index, d in enumerate(syn_data['train']['text']):
            try:
                if not d.startswith("Business Category: "):
                    synthetic_data.append(d)
            except:
                continue
    elif dataset == "openreview" or dataset == "pubmed":
        for index, d in enumerate(syn_data['train']['text']):
            len_d = num_tokens_from_string(d, encoding)
            if len_d > min_token_threshold:
                synthetic_data.append(d)
    elif dataset == "mimic":
        synthetic_data = [d for d in syn_data['train']['text']]
    else:
        synthetic_data = [d for d in syn_data['train']['text']]
    print("--- syn data len %d  ---" % (len(synthetic_data)))

    start_time = time.time()
    with torch.no_grad():
        synthetic_embeddings = []
        for i in tqdm(range(len(synthetic_data) // batch_size+1)):
            embeddings = model.encode(
                synthetic_data[i * batch_size:(i + 1) * batch_size])
            synthetic_embeddings.append(embeddings)
        all_synthetic_embeddings = np.concatenate(synthetic_embeddings)

    print("--- %s seconds for computing emb ---" % (time.time() - start_time))

    fid = calculate_fid(all_original_embeddings, all_synthetic_embeddings)
    print('FID : %.3f' % fid, len(all_original_embeddings),
          len(all_synthetic_embeddings))

    all_run_results = []

    for run in range(num_run):
        if (private_data_size != -1) and (len(all_original_embeddings) > private_data_size):
            rand_index = np.random.choice(
                list(range(len(all_original_embeddings))), size=private_data_size, replace=False)
            original_embeddings = all_original_embeddings[rand_index]
        else:
            original_embeddings = all_original_embeddings
        print("pri emb len", len(original_embeddings))

        if (private_data_size != -1) and (len(all_synthetic_embeddings) > private_data_size):
            rand_index = np.random.choice(list(
                range(len(all_synthetic_embeddings))), size=private_data_size, replace=False)
            synthetic_embeddings = all_synthetic_embeddings[rand_index]
        else:
            synthetic_embeddings = all_synthetic_embeddings
        print("syn emb len", len(synthetic_embeddings))

        start_time = time.time()
        precision, recall, f1, mauve, kl, tv, wass, sinkhorn_loss = calculate_all_metrics(
            synthetic_embeddings, original_embeddings, k)
        
        if real_file != "":
            df = pd.read_csv(real_file)  
            real_text_list = df["text"].tolist()

            # bert_score = compute_bertscore(real_text_list, synthetic_data)
            bert_score = compute_bertscore(real_text_list[:len(synthetic_data)], synthetic_data)["f1"]
            # bert_score = compute_bertscore_pairwise(real_text_list, synthetic_data)["pairwise_f1_mean"]
            blue = compute_bleu(real_text_list, synthetic_data)

            self_blue = compute_self_bleu(synthetic_data)
            distinct_2 = compute_distinct_2(synthetic_data)

            real_lengths = get_lengths(real_text_list)
            synth_lengths = get_lengths(synthetic_data)

            plots_folder = os.path.join(synthetic_folder, "plots_metrics")
            os.makedirs(plots_folder, exist_ok=True)
            plot_length_distributions(real_lengths, synth_lengths, filename=os.path.join(plots_folder, f"length_distribution_{epoch}.png"))
            dict_lengths_metrics = compare_length_distributions(real_lengths, synth_lengths)

        else:
            bert_score, blue, self_blue, distinct_2 = 0, 0, 0, 0

        print("--- %s seconds for computing metric ---" %
              (time.time() - start_time))

        with open(csv_fname, 'a', newline='') as file:
            writer = csv.writer(file)
            if run == 0:
                writer.writerow(["run", "fid", "precision", "recall",
                                "f1", "mauve", "kl", "tv", "wass", "sinkhorn_loss",
                                "bert_score", "blue", "self_blue", "distinct_2",
                                "length_real_mean",
                                "length_synthetic_mean",
                                "length_real_std",
                                "length_synthetic_std",
                                "length_kl_divergence",
                                "length_wasserstein_distance"])
            row_list = [
                round(fid, 4),
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                round(mauve, 4),
                round(kl, 4),
                round(tv, 4),
                round(wass, 4),
                round(sinkhorn_loss, 4),

                round(bert_score, 4),
                round(blue, 4),
                round(self_blue, 4),
                round(distinct_2, 4),

                round(dict_lengths_metrics["length_real_mean"], 4),
                round(dict_lengths_metrics["length_synthetic_mean"], 4),
                round(dict_lengths_metrics["length_real_std"], 4),
                round(dict_lengths_metrics["length_synthetic_std"], 4),
                round(dict_lengths_metrics["length_kl_divergence"], 4),
                round(dict_lengths_metrics["length_wasserstein_distance"], 4),
            ]
            writer.writerow([run]+row_list)

        all_run_results.append(row_list)

    mean_run_results = np.mean(np.array(all_run_results), axis=0).tolist()
    mean_run_results = [round(x, 4) for x in mean_run_results]
    with open(csv_fname, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["avg"] + mean_run_results)
    
    # Create metrics dictionary
    metrics_dict = {
        "fid": mean_run_results[0],
        "precision": mean_run_results[1],
        "recall": mean_run_results[2],
        "f1": mean_run_results[3],
        "mauve": mean_run_results[4],
        "kl": mean_run_results[5],
        "tv": mean_run_results[6],
        "wass": mean_run_results[7],
        "sinkhorn_loss": mean_run_results[8],

        "bert_score": mean_run_results[9],
        "blue": mean_run_results[10],
        "self_blue": mean_run_results[11],
        "distinct_2": mean_run_results[12],

        "length_real_mean": mean_run_results[13],
        "length_synthetic_mean": mean_run_results[14],
        "length_real_std": mean_run_results[15],
        "length_synthetic_std": mean_run_results[16],
        "length_kl_divergence": mean_run_results[17],
        "length_wasserstein_distance": mean_run_results[18],

    }

    if wandb.run is not None:
        wandb.log({"epoch": epoch, **{f"metrics_1/{k}": v for k, v in metrics_dict.items()}})
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"epoch": epoch, **metrics_dict})
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_file", type=str,
                       default="", required=False)
    parser.add_argument(
        '--train_data_embeddings_file',
        type=str,
        default="")

    parser.add_argument("--synthetic_file", type=str,
                       default="",
                       required=False)
    parser.add_argument("--synthetic_folder", type=str,
                       default="",
                       required=False)
    parser.add_argument("--synthetic_iteration", type=int,
                       default=20,
                       required=False)
    parser.add_argument("--synthetic_start_iter", type=int,
                       default=0,
                       required=False)
    parser.add_argument("--min_token_threshold", type=int,
                       default=100,
                       required=False)
    
    parser.add_argument("--real_path", type=str,
                       default="data/mimic/train.csv",
                       required=False)

    parser.add_argument("--model_name_or_path", type=str,
                       default="stsb-roberta-base-v2", required=False)
    parser.add_argument("--metric", type=str, default="fid")
    parser.add_argument("--batch_size", type=int, required=False, default=1024)
    parser.add_argument("--private_data_size", type=int,
                       required=False, default=5000)
    parser.add_argument("--k", type=int, required=False, default=3)
    parser.add_argument("--run", type=int, required=False, default=1)
    parser.add_argument("--dataset", type=str, default="yelp",
                       choices=["yelp", "pubmed", "openreview", "mimic"],
                       required=False)
    parser.add_argument("--wandb_project", type=str, default="synthetic_data_evaluation",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--wandb_notes", type=str, default="",
                       help="Weights & Biases run notes")

    args = parser.parse_args()
    set_seed(seed=0, n_gpu=1)
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, 
               name=args.wandb_name,
               notes=args.wandb_notes,
               config=vars(args))
    
    model = SentenceTransformer(args.model_name_or_path)
    model.eval()

    dataset2embedding_file = {
        "yelp": f"result/embeddings/{args.model_name_or_path}/yelp_train_all.embeddings.npz",
        "pubmed": f"result/embeddings/{args.model_name_or_path}/pubmed_train_all.embeddings.npz",
        "openreview": f"result/embeddings/{args.model_name_or_path}/openreview_train_all.embeddings.npz",
        "mimic": f"result/embeddings/{args.model_name_or_path}/mimic_train_all.embeddings.npz",
    }
    args.train_data_embeddings_file = dataset2embedding_file[args.dataset]

    all_original_embeddings, original_labels = load_embeddings(
        args.train_data_embeddings_file)

    if args.private_data_size == -1:
        args.run = 1

    metrics_history = {}  # To store metrics across epochs
    
    if args.synthetic_folder == '':
        if args.synthetic_file != '':
            csv_fname = os.path.join(os.path.dirname(
                args.synthetic_file), 'eval_metric.csv')
            metrics = eval_one_file(syn_fname=args.synthetic_file, 
                                  all_original_embeddings=all_original_embeddings, 
                                  model=model,
                                  csv_fname=csv_fname, 
                                  batch_size=args.batch_size,
                                  private_data_size=args.private_data_size,
                                  num_run=args.run, 
                                  k=args.k, 
                                  dataset=args.dataset,  
                                  min_token_threshold=args.min_token_threshold,
                                  epoch=0)
            metrics_history[0] = metrics
    else:
        for _iter in range(args.synthetic_start_iter, args.synthetic_iteration + 1):
            print("\n______________________________\n", "ITERATION !!!!!!!", _iter, "\n______________________________\n")
            syn_data_file = os.path.join(
                args.synthetic_folder, str(_iter), 'samples.csv')
            if os.path.isfile(syn_data_file):
                csv_fname = os.path.join(
                    args.synthetic_folder, str(_iter), 'eval_metric.csv')
                # if os.path.exists(csv_fname):
                #     # Load existing metrics if file exists
                #     with open(csv_fname, 'r') as f:
                #         reader = csv.reader(f)
                #         rows = list(reader)
                #         if len(rows) > 1 and rows[-1][0] == "avg":
                #             metrics = {
                #                 "fid": float(rows[-1][1]),
                #                 "precision": float(rows[-1][2]),
                #                 "recall": float(rows[-1][3]),
                #                 "f1": float(rows[-1][4]),
                #                 "mauve": float(rows[-1][5]),
                #                 "kl": float(rows[-1][6]),
                #                 "tv": float(rows[-1][7]),
                #                 "wass": float(rows[-1][8]),
                #                 "sinkhorn_loss": float(rows[-1][9]),
                #             }
                #             metrics_history[_iter] = metrics
                #     continue
                
                print(f'Processing {csv_fname}')
                metrics = eval_one_file(syn_fname=syn_data_file, 
                                      all_original_embeddings=all_original_embeddings, 
                                      model=model,
                                      csv_fname=csv_fname, 
                                      batch_size=args.batch_size,
                                      private_data_size=args.private_data_size,
                                      num_run=args.run, 
                                      k=args.k, 
                                      dataset=args.dataset, 
                                      min_token_threshold=args.min_token_threshold,
                                      epoch=_iter, real_file=args.real_path, synthetic_folder=args.synthetic_folder)
                metrics_history[_iter] = metrics
            else:
                print(f"{syn_data_file} does not exist")
    
    # After processing all epochs, plot metrics
    if metrics_history:
        plot_dir = os.path.join(args.synthetic_folder, "metrics_plots") if args.synthetic_folder else os.path.join(os.path.dirname(args.synthetic_file), "metrics_plots")
        plot_metrics(metrics_history, plot_dir)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()