mlm_prob=0.6
var_type="mimic_rephrase_tone"
feat_ext="sentence-t5-base"
length=1024
temperature=1.0
num_seed_samples=10
lookahead_degree=0
k=6 # number of variations
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=10
word_var_scale=0
select_syn_mode=rank
random_str="_ollama_summary_01"
percentage_of_summaries=0.5
# model_type="gpt2"
# model_type="mistral"
# model_type="deepseek-v2.5"
model_type="aravhawk/llama4"
# model_type="ingu627/llama4-scout-q4:109b"
# model_type="llama3.3"
# model_type="deepseek-r1:70b"
# model_type="qwen:72b"
# model_type="mistral-small3.1"
# model_type="mistral-large"
# model_type="qwen3:235b"
# model_type="deepseek-v2:16b"
# model_type="mistralai/Mixtral-8x7B-Instruct-v0.1"
noise=0
args=""
api="HFGPT"
feature_extractor_batch_size=1024
if [ "$model_type" = "gpt2-large" ]; then
    batch_size=32
elif [ "$model_type" = "gpt2-medium" ]; then
    batch_size=64
elif [ "$model_type" = "gpt2" ]; then
    batch_size=128
else
    batch_size=16
fi
result_folder="result/mimic/${model_type}_${feat_ext}/${percentage_of_summaries}_${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}_t${temperature}_${random_str}"


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

pip install bert_score

### run PE
python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/mimic/train.csv" \
--dataset "mimic" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
--percentage_of_summaries ${percentage_of_summaries} \
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
--log_online \
# --train_data_embeddings_file "result/embeddings/${feat_ext}/mimic_train_all.embeddings.npz" 

