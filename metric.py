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

from dpsda.logging import *
from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *
from apis.utils import set_seed

import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_string(string, encoding):
    """Returns the number of tokens in a text string."""
    try:
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens


# calculate frechet inception distance
def calculate_fid(act1, act2):
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


def eval_one_file(syn_fname, all_original_embeddings, model, csv_fname, batch_size, private_data_size, num_run, k, dataset="yelp", min_token_threshold=100, epoch=None):
    syn_data = load_dataset("csv", data_files=syn_fname)

    synthetic_data = []
    if dataset == "yelp":
        for index, d in enumerate(syn_data['train']['text']):
            try:
                if not d.startswith("Business Category: "):
                    synthetic_data.append(d)
            except:
                continue
    elif dataset == "openreview" or dataset == "pubmed" or dataset == "mimic":
        for index, d in enumerate(syn_data['train']['text']):
            len_d = num_tokens_from_string(d, encoding)
            if len_d > min_token_threshold:
                synthetic_data.append(d)
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

        print("--- %s seconds for computing metric ---" %
              (time.time() - start_time))

        with open(csv_fname, 'a', newline='') as file:
            writer = csv.writer(file)
            if run == 0:
                writer.writerow(["run", "fid", "precision", "recall",
                                "f1", "mauve", "kl", "tv", "wass", "sinkhorn_loss"])
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
            syn_data_file = os.path.join(
                args.synthetic_folder, str(_iter), 'samples.csv')
            if os.path.isfile(syn_data_file):
                csv_fname = os.path.join(
                    args.synthetic_folder, str(_iter), 'eval_metric.csv')
                if os.path.exists(csv_fname):
                    # Load existing metrics if file exists
                    with open(csv_fname, 'r') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        if len(rows) > 1 and rows[-1][0] == "avg":
                            metrics = {
                                "fid": float(rows[-1][1]),
                                "precision": float(rows[-1][2]),
                                "recall": float(rows[-1][3]),
                                "f1": float(rows[-1][4]),
                                "mauve": float(rows[-1][5]),
                                "kl": float(rows[-1][6]),
                                "tv": float(rows[-1][7]),
                                "wass": float(rows[-1][8]),
                                "sinkhorn_loss": float(rows[-1][9]),
                            }
                            metrics_history[_iter] = metrics
                    continue
                
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
                                      epoch=_iter)
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