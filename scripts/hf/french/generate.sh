mlm_prob=0.6
var_type="french_rephrase_tone"
# feat_ext="UFNLP/gatortron-base"
# feat_ext="stsb-roberta-base-v2"
# feat_ext="all-mpnet-base-v2"
# feat_ext="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# feat_ext="Qwen/Qwen3-Embedding-8B"
# feat_ext="abhinand/MedEmbed-large-v0.1"
feat_ext="models2/french_downloading/model_emb_qwen"

length=512
temperature=1.0
num_seed_samples=20
lookahead_degree=0
k=2 # number of variations
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=10
word_var_scale=0
select_syn_mode=rank
random_str="exp_07_19"

percentage_of_summaries=0.0
summaries_model="without_summ"
# summaries_model="llama4"
summaries_path="data/french/summarized_texts_${summaries_model}.csv"
diversity_percentage=0.0
length_mean=100
length_std=10
length_min=20
length_max=512

local_model_path="models2/french_downloading/mistral-7b-instruct-v0.3"

model_type="${local_model_path}"

# model_type="deepseek-v2.5:latest"
# model_type="deepseek-v2:16b"
# model_type="llama3.3:latest"
# model_type="llama4:16x17b"
# model_type="qwen:72b"
# model_type="qwen3:32b"
# model_type="mistral-small3.1:latest"
# model_type="mixtral:latest"
# model_type="mistral-large:latest"
# model_type="gemma3:latest"
# model_type="gemma3:27b"

model_type_2="${model_type}"
if [[ "$model_type" == *:* ]]; then
    model_type_2="${model_type//:/_}"
fi

noise=0
args=""
api="HFGPT"
feature_extractor_batch_size=1
if [ "$model_type" = "gpt2-large" ]; then
    batch_size=32
elif [ "$model_type" = "gpt2-medium" ]; then
    batch_size=64
elif [ "$model_type" = "gpt2" ]; then
    batch_size=128
else
    batch_size=16
fi

# mimic_summ_diversity_embed
result_folder="result/french_start/${model_type_2}_${feat_ext}/perc_of_summ_${percentage_of_summaries}_summ_mod_${summaries_model}_divers_perc_${diversity_percentage}_len_mean${length_mean}_len_std${length_std}_${num_samples}_n${noise}_L${L}_t${temperature}_${random_str}"


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

### run PE
CUDA_VISIBLE_DEVICES=1,2,3 python main.py ${args} ${data_checkpoint_args} \
--train_data_file "../../data/french_data.csv" \
--dataset "french" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
--percentage_of_summaries ${percentage_of_summaries} \
--summaries_path ${summaries_path} \
--diversity_percentage ${diversity_percentage} \
--length_mean ${length_mean} \
--local_model_path ${local_model_path} \
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
--train_data_embeddings_file "result/embeddings/${feat_ext}/french_train_all.embeddings.npz" 

