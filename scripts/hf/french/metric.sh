
# feat_ext="sentence-t5-base"
# feat_ext="Qwen/Qwen3-Embedding-8B"
# feat_ext="abhinand/MedEmbed-large-v0.1"
# feat_ext="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
feat_ext="models2/french_downloading/model_emb_qwen"
epochs=10
model_folder="gemma3_latest_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.0_len_mean2487_len_std930_90_n5_L3_t1.0_exp_07_05"

# mistral-small3.1_latest_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.1_len_mean2487_len_std930_90_n0_L3_t1.0_exp_07_03
# mixtral_latest_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.0_len_mean2487_len_std930_90_n0_L3_t1.0_exp_07_01
# gemma3_27b_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.0_len_mean2487_len_std930_90_n0_L3_t1.0_exp_07_03
# gemma3_latest_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.0_len_mean2487_len_std930_90_n5_L3_t1.0_exp_07_05
# llama3.3_Qwen/Qwen3-Embedding-8B/perc_of_summ_0.0_summ_mod_without_summ_divers_perc_0.0_len_mean2487_len_std930_90_n0_L3_t1.0_exp_07_03


result_folder="result/french/${model_folder}"
min_token_threshold=50

# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.7_summ_mod_llama4_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.7_summ_mod_deepseek-v2.5_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_llama4_divers_numb_30_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_llama4_divers_numb_6_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_llama4_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# llama4:scout_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_deepseek-v2.5_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11


# mistral-small3.1_stsb-roberta-base-v2/perc_of_summ_0.7_summ_mod_llama4_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# mistral-small3.1_stsb-roberta-base-v2/perc_of_summ_0.7_summ_mod_deepseek-v2.5_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11

# mistral-small3.1_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_llama4_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11
# mistral-small3.1_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_deepseek-v2.5_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11


CUDA_VISIBLE_DEVICES=0 python metric.py \
    --private_data_size 100 \
    --synthetic_folder ${result_folder} \
    --run 1  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${epochs} \
    --original_file "data/mimic/train.csv"  \
    --train_data_embeddings_file result/embeddings/${feat_ext}/mimic_train_all.embeddings.npz \
    --model_name_or_path ${feat_ext} \
    --dataset "mimic" \
    --wandb_name ${model_folder} \