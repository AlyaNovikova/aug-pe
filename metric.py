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
from collections import defaultdict
import pandas as pd

# calculate inception score with Keras
import torch
import argparse
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset

from dpsda.metrics import calculate_fid, num_tokens_from_string, get_lengths, plot_length_distributions, plot_metrics

from dpsda.logging import *
from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *
from apis.utils import set_seed 

import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

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

    # all_run_results = []
    all_run_results = defaultdict(list)

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
        # precision, recall, f1, mauve, kl, tv, wass, sinkhorn_loss = calculate_all_metrics(
        #     original_embeddings, synthetic_embeddings, k)
        
        if real_file != "":
            df = pd.read_csv(real_file)  
            real_text_list = df["text"].tolist()

            metrics = compare_text_sets(real_text_list, synthetic_data, original_embeddings, synthetic_embeddings)

            real_lengths = get_lengths(real_text_list)
            synth_lengths = get_lengths(synthetic_data)
            plots_folder = os.path.join(synthetic_folder, "plots_metrics")
            os.makedirs(plots_folder, exist_ok=True)
            plot_length_distributions(real_lengths, synth_lengths, filename=os.path.join(plots_folder, f"length_distribution_{epoch}.png"))


        else:
            metrics = {}


        print("--- %s seconds for computing metric ---" %
              (time.time() - start_time))

        metric_keys = list(metrics.keys())  
        
        with open(csv_fname, 'a', newline='') as file:
            writer = csv.writer(file)

            if run == 0:
                writer.writerow(["run"] + metric_keys)
            
            row = [round(metrics.get(k, 0), 4) for k in metric_keys]
            writer.writerow([run] + row)

        for k in metric_keys:
            all_run_results[k].append(metrics.get(k, 0))

    metrics_dict = {}
    mean_run_results = []

    with open(csv_fname, 'a', newline='') as file:
        writer = csv.writer(file)

        avg_row = []
        for k in metric_keys:
            mean_val = round(np.mean(all_run_results[k]), 4)
            metrics_dict[k] = mean_val
            avg_row.append(mean_val)
        
        writer.writerow(["avg"] + avg_row)

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
    parser.add_argument("--wandb_project", type=str, default="synthetic_data_evaluation_check",
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