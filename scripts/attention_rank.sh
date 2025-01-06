#!/bin/bash

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name_2="dellaanima/KE_Meta-Llama-3-8B-Instruct_MEMIT_CF5000"


data_dir="../data/final_5000.json"
data_dir_2="../data/real_final_e_full_with_generated_sentences.json"
log_dir="../logs/attention_rank.log"

cache_dir="../../.cache"
cuda_num="0"


python ../analysis/Attention/attention_rank.py --model_name "$model_name" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir_2"
python ../analysis/Attention/attention_rank.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir_2"
