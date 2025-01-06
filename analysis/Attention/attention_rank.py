import torch
import torch
import sys
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np

def get_max_attention_token(text, tokenizer, model, k=0, block_num=-1):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    
    # Check token split
    
    token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    if block_num != -1:
        attention_mask = inputs['attention_mask']
        attention_mask[:, block_num] = 0
        inputs['attention_mask'] = attention_mask
    
    outputs = model(**inputs)
    attention_values = outputs.attentions

    sequence_length = inputs['input_ids'].shape[1]
    num_layers = len(attention_values)
    
    # 히트맵 데이터 초기화
    heatmap_data = torch.zeros((num_layers, sequence_length - k))

    # 히트맵 데이터 계산
    for layer_index, attention_layer in enumerate(attention_values):
        attention_layer_mean = attention_layer.mean(dim=1)  # (batch_size, sequence_length, sequence_length)
        attention_layer_mean = attention_layer_mean.squeeze(0)  # (sequence_length, sequence_length)
        attention_to_last_position = attention_layer_mean[-1]  # (sequence_length,)
        heatmap_data[layer_index, :] = attention_to_last_position[k:]

    # 전체 레이어에 대해 평균 계산 (축소)
    average_attention = heatmap_data.mean(dim=0)  # (sequence_length - k)

    # NumPy로 변환 및 마지막 마침표 위치까지 슬라이싱
    average_attention_np = average_attention.detach().cpu().numpy()
    
    # 마지막 마침표 위치 찾기
    last_period_index = len(token_list) - 1  # 기본값: 끝까지
    for i, token in enumerate(token_list):
        if token == '.':  # 마지막 마침표를 찾음
            last_period_index = i

    # 마지막 마침표와 해당 attention 값 제거
    average_attention_np = average_attention_np[:last_period_index - k]  # 마지막 마침표 제외
    cleaned_tokens = [token.replace('Ġ', '') for token in token_list[k:last_period_index]]  # 마지막 마침표 제외

    # 가장 큰 평균값을 가지는 토큰 찾기
    max_index = np.argmax(average_attention_np)  # 최대값의 인덱스
    max_token = cleaned_tokens[max_index]  # 최대값에 해당하는 토큰
    max_value = average_attention_np[max_index]  # 최대값

    return max_token, max_value

def get_max_attention_token_user_text(text, tokenizer, model,  block_num=-1):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # Extract token list
    token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Identify the start and end indices of the user section
    user_start_tokens = ["<|start_header_id|>", "user", "<|end_header_id|>"]
    user_end_token = "<|eot_id|>"

    # Find start index of user text
    try:
        user_start_idx = None
        for i in range(len(token_list) - len(user_start_tokens) + 1):
            if token_list[i:i + len(user_start_tokens)] == user_start_tokens:
                user_start_idx = i + len(user_start_tokens)
                break

        if user_start_idx is None:
            raise ValueError("User start marker not found in token list.")

        # Find end index of user text
        user_end_idx = token_list.index(user_end_token, user_start_idx)
    except ValueError:
        raise ValueError("Could not find user text markers in the provided input.")
    
    # Focus only on user text tokens
    user_tokens = token_list[user_start_idx+1:user_end_idx-1]

    # Get model attention outputs
    model.config.output_attentions = True
    model.config.output_hidden_states = True

    if block_num != -1:
        attention_mask = inputs['attention_mask']
        attention_mask[:, block_num] = 0
        inputs['attention_mask'] = attention_mask

    outputs = model(**inputs)
    attention_values = outputs.attentions

    # Get sequence length and number of layers
    sequence_length = inputs['input_ids'].shape[1]
    num_layers = len(attention_values)

    # Initialize heatmap data
    heatmap_data = torch.zeros((num_layers, sequence_length))

    for layer_index, attention_layer in enumerate(attention_values):
        attention_layer_mean = attention_layer.mean(dim=1).squeeze(0)
        attention_to_last_position = attention_layer_mean[-1]
        heatmap_data[layer_index, :] = attention_to_last_position[:]

    # Calculate average attention over layers
    average_attention = heatmap_data.mean(dim=0)
    average_attention_np = average_attention.detach().cpu().numpy()

    # Focus only on user tokens within the range of the token list
    user_attention = average_attention_np[user_start_idx+1:user_end_idx-1] # exclude "."
    user_tokens_cleaned = [token.replace('Ġ', '') for token in user_tokens]

    # Find the token with the highest attention value
    max_index = np.argmax(user_attention)
    max_token = user_tokens_cleaned[max_index]
    max_value = user_attention[max_index]

    return max_token, max_value

def make_chat_temp(hop_sentence, question):
    char = [
        {"role": "user", "content": hop_sentence},
    ]
    text = tokenizer.apply_chat_template(char, tokenize = False) + '<|start_header_id|>assistant<|end_header_id|>\n\n' + question
    text = text.replace("<|begin_of_text|>", "", 1)
    #text = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + hop_sentence + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + question
    return text

def sbj(data, opt = 0):
    k=0
    count = 0
    count2 = 0
    for i in range(len(data)):
        text = data[i]['generated_sentences']['sbj_hop_test']["sentence_with_hop_and_original"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
        sbj = data[i]['subject']
        sbj2 = data[i]['sbj_hop_test'][k]
        word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, max_length=150, block_num = -1)
        if word in sbj:
            count += 1
        elif word in sbj2:
            count2 += 1
    return count, count2

def sbj_hop(data, opt = 0):
    k=0
    count = 0
    for i in tqdm(range(len(data))):
        sbj = data[i]['sbj_hop_test'][k]
        if opt == 0:
            text = data[i]['generated_sentences']['sbj_hop_test']["sentence_with_hop_word"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
            word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, block_num = -1)
        else:
            text = make_chat_temp(data[i]['generated_sentences']['sbj_hop_test']["sentence_with_hop_word"][k], data[i]['prompt'].format(data[i]['subject']))
            word, attention_score = get_max_attention_token_user_text(text, tokenizer, model, block_num = -1)
        if word in sbj:
            count += 1
    return count

def obj(data):
    k = 0
    count = 0
    count2 = 0
    for i in range(len(data)):
        text = data[i]['generated_sentences']['obj_true_hop_test']["sentence_with_hop_and_original"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
        sbj = data[i]['fact_knowledge']
        sbj2 = data[i]['obj_true_hop_test'][k]
        word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, max_length=150, block_num = -1)
        #if sbj.endswith(word):
        #    count += 1
        if word in sbj:
            count += 1
        elif word in sbj2:
            count2 += 1
    return count, count2    

def obj_hop(data, opt = 0):
    k=0
    count = 0
    for i in tqdm(range(len(data))):
        sbj = data[i]['obj_true_hop_test'][k]
        if opt == 0:
            text = data[i]['generated_sentences']['obj_true_hop_test']["sentence_with_hop_word"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
            word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, block_num = -1)
        else:
            text = make_chat_temp(data[i]['generated_sentences']['obj_true_hop_test']["sentence_with_hop_word"][k], data[i]['prompt'].format(data[i]['subject']))
            word, attention_score = get_max_attention_token_user_text(text, tokenizer, model, block_num = -1)
        
        if word in sbj:
            count += 1
    return count

def obj_new(data):
    k = 0
    count = 0
    count2 = 0
    for i in range(len(data)):
        text = data[i]['generated_sentences']['obj_new_hop_test']["sentence_with_hop_and_original"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
        sbj = data[i]['edited_knowledge']
        sbj2 = data[i]['obj_new_hop_test'][k]
        word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, max_length=150, block_num = -1)
        if word in sbj:
            count += 1
        elif word in sbj2:
            count2 += 1
    return count, count2   

def obj_new_hop(data, opt = 0):
    k=0
    count = 0
    for i in tqdm(range(len(data))):
        
        sbj = data[i]['obj_new_hop_test'][k]
        if opt == 0:
            text = data[i]['generated_sentences']['obj_new_hop_test']["sentence_with_hop_word"][k] + ' ' + data[i]['prompt'].format(data[i]['subject'])
            word, attention_score = get_max_attention_token(text, tokenizer, model, k=1, block_num = -1)
        else:
            text = make_chat_temp(data[i]['generated_sentences']['obj_new_hop_test']["sentence_with_hop_word"][k], data[i]['prompt'].format(data[i]['subject']))
            word, attention_score = get_max_attention_token_user_text(text, tokenizer, model, block_num = -1)

        if word in sbj:
            count += 1
    return count

def total(data, opt = 0):
    #s1, s2 = sbj(data)
    s3 = sbj_hop(data, opt)
    logging.info(f'Subject hop sentence {s3}')
    #o1, o2 = obj(data)
    o3 = obj_hop(data, opt)
    logging.info(f'Object true hop sentence {o3}')
    #n1, n2 = obj_new(data)
    n3 = obj_new_hop(data, opt)
    logging.info(f'Object new hop sentence {n3}')
    #ans = [s1, s2, s3, o1, o2, o3, n1, n2, n3]
    ans = [s3, o3, n3]
    return ans



if __name__ == "__main__":
        


    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from utils.setting import set_seed, now

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default = "meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--log_dir', type=str, default = "experiment.log")
    parser.add_argument('--cache_dir', type=str, default = "../../.cache")
    parser.add_argument('--data_dir', type=str, default = 'df_exp1_e_5000_with_generated_sentences.json')
    parser.add_argument('--cuda_num', type=str, default = 0)

    

    args = parser.parse_args()
    model_name = args.model_name
    cache_dir = args.cache_dir
    data = args.data_dir
    cuda_num = args.cuda_num
    log_dir = args.log_dir
    
    # 로깅 설정: 'app.log' 파일에 INFO 수준 이상의 로그를 저장
    logging.basicConfig(
        filename=log_dir,                     # 로그 파일 이름 지정
        filemode='a',                           # 파일 모드 ('a'는 append, 'w'는 overwrite)
        level=logging.INFO,                     # 로그 수준 설정
        format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 형식 지정
    )
    
    logging.info(f'model_name : {model_name}')
    logging.info(f'data : {data}')

    
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

    now()

    seed = 42
    set_seed(seed)


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir = cache_dir, torch_dtype=torch.float16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float16).to('cuda')

    with open(data, 'r') as file:
        data = json.load(file) 
        
    for i in range(0,2):
        opt = i
        if opt == 1:
            logging.info('Chat Template = True')
        else:
            logging.info('Chat Template = False')
        ans = total(data, opt = i)
        
