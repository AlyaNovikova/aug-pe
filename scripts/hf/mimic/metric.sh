
feat_ext="sentence-t5-base"
epochs=10
model_folder="mistral_small/70_n0_L7_initL7_var0_mimic_rephrase_tone_rank_len1024var0_t1.0__ollama_rephrase_pr_01"
result_folder="result/mimic_play_metric/${model_folder}"
min_token_threshold=50

pip install bert_score
pip install rouge-score

CUDA_VISIBLE_DEVICES=0 python metric.py \
    --private_data_size 100 \
    --synthetic_folder ${result_folder} \
    --run 1  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${epochs} \
    --original_file "data/mimic/train.csv"  \
    --train_data_embeddings_file result/embeddings/${feat_ext}/mimic_train_all.embeddings.npz \
    --model_name_or_path ${feat_ext} \
    --dataset mimic \
    --wandb_name ${model_folder} \