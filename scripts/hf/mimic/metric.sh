
feat_ext="stsb-roberta-base-v2"
epochs=13
model_folder="mistral-small3.1_stsb-roberta-base-v2/perc_of_summ_0.0_summ_mod_deepseek-v2.5_divers_numb_0_len_mean2487_len_std930_120_n0_L6_t1.0__parametrs_len_diversity_summ11"
result_folder="result/mimic_summ_diversity_embed/${model_folder}"
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
    --dataset mimic \
    --wandb_name ${model_folder} \