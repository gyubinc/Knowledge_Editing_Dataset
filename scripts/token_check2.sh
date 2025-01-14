#!/bin/bash

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name_2="dellaanima/KE-Data-1000-Edits_MEMIT-layer3_llama3-Instruct-8b"


data_dir="../data/final_2000.json"
log_dir="../logs/token_check_2222.log"

cache_dir="../../.cache"
cuda_num="2"

python ../analysis/Attention/token_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --chat_opt 0 --random_opt 0 --random_word 0
# python ../analysis/Attention/token_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 1 --chat_opt 0 --random_opt 0 --random_word 0
# python ../analysis/Attention/token_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --chat_opt 0 --random_opt 0 --random_word 1
# python ../analysis/Attention/token_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --chat_opt 0 --random_opt 1 --random_word 0
# python ../analysis/Attention/token_check.py --model_name "$model_name_2" --log_dir "$log_dir" --cache_dir "$cache_dir" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 1 --chat_opt 0 --random_opt 1 --random_word 0

