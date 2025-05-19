#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 --dataset_name"
  exit 1
fi

# Use a case statement to handle different datasets
case $1 in
  --openreview)
    python pre_comp_emb.py --dataset openreview --model_name_or_path 'stsb-roberta-base-v2'
    ;;
  --pubmed)
    python pre_comp_emb.py --dataset pubmed --model_name_or_path 'sentence-t5-base'
    ;;
  --mimic)
    # export CUDA_VISIBLE_DEVICES=0
    python pre_comp_emb.py --dataset mimic --model_name_or_path 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    ;;
  --yelp)
    python pre_comp_emb.py --dataset yelp --model_name_or_path 'stsb-roberta-base-v2'
    ;;
  *)
    echo "Invalid dataset. Available datasets are: --openreview, --pubmed, --mimic, --yelp"
    exit 1
    ;;
esac
