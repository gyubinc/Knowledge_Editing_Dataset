#!/bin/bash

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name_2="dellaanima/KE_Meta-Llama-3-8B-Instruct_MEMIT_CF5000"

data_dir="df_exp1_e_5000_with_generated_sentences.json"
log_name="token_check2.log"

cuda_num="1"

python token_check.py --model_name "$model_name_2" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0 --log_name "$log_name"
python token_check.py --model_name "$model_name_2" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 1 --log_name "$log_name"