mlm_prob=0.6
var_type="pubmed_rephrase_tone"
feat_ext="sentence-t5-base"
length=448
temperature=1.0
num_seed_samples=10
lookahead_degree=0
k=6 # number of variations
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=2
word_var_scale=0
select_syn_mode=rank
model_type=gpt2
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
    batch_size=8
fi
result_folder="result/pubmed_mimic/${model_type}_${feat_ext}/${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}_t${temperature}"


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
python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/mimic/train.csv" \
--dataset "pubmed" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
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
--train_data_embeddings_file "result/embeddings/${feat_ext}/pubmed_train_all.embeddings.npz" 

