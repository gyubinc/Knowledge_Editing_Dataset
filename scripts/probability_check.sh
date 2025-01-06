#!/bin/bash

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name_2="dellaanima/KE_Meta-Llama-3-8B-Instruct_MEMIT_CF5000"

data_dir="../data/final_5000.json"
log_dir="../logs/probability_check.log"

cache_dir="../../.cache"
cuda_num="0"

python ../analysis/Attention/probability_check.py --model_name "$model_name" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --chat_opt 1
python ../analysis/Attention/probability_check.py --model_name "$model_name" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 1 --chat_opt 1
python ../analysis/Attention/probability_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --chat_opt 1
python ../analysis/Attention/probability_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 1 --chat_opt 1


