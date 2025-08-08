mlm_prob=0.6
var_type="mimic_rephrase_tone"
folder_for_save="final2_fixed_divers_embed_and_noise"
random_str="exp_07_29"

feat_ext="Qwen/Qwen3-Embedding-8B"
# feat_ext="abhinand/MedEmbed-large-v0.1"

num_seed_samples=36
k=2 # number of variations
few_shot="No"
self_refinement="No"
multi_models="No"
bad_prompts="No"
noise=2.4
temperature=1.0
diversity_percentage=0.0

# model_type="mixtral:latest"
# model_type="qwen:72b"
model_type="mistral-small3.1:latest"
# model_type="llama3.3:latest"

percentage_of_summaries=0.0
summaries_model="without_summ"
# summaries_model="llama4"

# model_type="deepseek-v2.5:latest"
# model_type="deepseek-v2:16b"
# model_type="llama4:16x17b"
# model_type="qwen3:32b"
# model_type="mistral-large:latest"
# model_type="gemma3:latest"
# model_type="gemma3:27b"

model_one="qwen:72b"
model_two="mistral-small3.1:latest"

length=2048
lookahead_degree=0
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=13
word_var_scale=0
select_syn_mode=rank
length_mean=2487
length_std=930
length_min=1200
length_max=4000
summaries_path="data/mimic/summarized_texts_${summaries_model}.csv"

model_type_2="${model_type}"
if [[ "$model_type" == *:* ]]; then
    model_type_2="${model_type//:/_}"
fi

model_type_3="${model_one}"
if [[ "$model_one" == *:* ]]; then
    model_type_3="${model_one//:/_}"
fi

model_type_4="${model_two}"
if [[ "$model_two" == *:* ]]; then
    model_type_4="${model_two//:/_}"
fi

args=""
api="HFGPT"
feature_extractor_batch_size=1
batch_size=16

result_folder="result/${folder_for_save}/${model_type_2}_${feat_ext}/models_${model_type_3}_${model_type_4}_summ_${percentage_of_summaries}_${summaries_model}_divers_${diversity_percentage}_${num_samples}_nois${noise}_L${L}_t${temperature}_few${few_shot}_self${self_refinement}_mult${multi_models}_bad${bad_prompts}_${random_str}"
wandb_name_s="${model_type_2}_${num_samples}_div_${diversity_percentage}_ns${noise}_L${L}_t${temperature}_few${few_shot}_self${self_refinement}_mult${multi_models}_bad${bad_prompts}_${feat_ext}_${model_type_3}_${model_type_4}_summ_${percentage_of_summaries}_${summaries_model}_${random_str}"

### load datacheckpoint 
data_checkpoint_args=""
for  (( iter=0; iter<=epochs; iter++ ))
do
train_file=${result_folder}/${iter}/samples.csv
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    # load from  data checkpoint
    data_checkpoint_args="--data_checkpoint_step ${iter} --data_checkpoint_path ${result_folder}/${iter}/samples.csv"
else
    echo "$train_file does not exist."
fi
done
echo load data from ${data_checkpoint_args} ${args}


pip install textstat
pip install gensim

### run PE
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/mimic/train.csv" \
--dataset "mimic" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
--percentage_of_summaries ${percentage_of_summaries} \
--summaries_path ${summaries_path} \
--diversity_percentage ${diversity_percentage} \
--length_mean ${length_mean} \
--length_std ${length_std} \
--length_max ${length_max} \
--length_min ${length_min} \
--do_sample  \
--length ${length} \
--random_sampling_batch_size ${batch_size} \
--variation_batch_size ${batch_size} \
--fp16 \
--temperature ${temperature} \
--select_syn_mode ${select_syn_mode} \
--num_samples_schedule ${num_samples} \
--combine_divide_L ${L} \
--init_combine_divide_L ${init_L} \
--variation_degree_schedule ${mlm_prob} \
--lookahead_degree ${lookahead_degree} \
--feature_extractor_batch_size ${feature_extractor_batch_size} \
--epochs ${epochs} \
--use_subcategory \
--feature_extractor ${feat_ext} \
--mlm_probability ${mlm_prob} \
--variation_type ${var_type} \
--result_folder ${result_folder} \
--wandb_name_s ${wandb_name_s} \
--log_online \
--train_data_embeddings_file "result/embeddings/${feat_ext}/mimic_train_all.embeddings.npz" \
--few_shot ${few_shot} \
--self_refinement ${self_refinement} \
--multi_models ${multi_models} \
--model_one ${model_one} \
--model_two ${model_two} \
--bad_prompts ${bad_prompts}
