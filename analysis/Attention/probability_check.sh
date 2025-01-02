#!/bin/bash

# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="dellaanima/KE_Meta-Llama-3-8B-Instruct_MEMIT_CF5000"

data_dir="df_exp1_e_5000_with_generated_sentences.json"

cuda_num="0"

python probability_check.py --model_name "$model_name" --cuda_num "$cuda_num" --data_dir "$data_dir" --block_opt 0
