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



def compute_sequence_blocking_probability(model, tokenizer, prefix, target, block_text, block_opt = 0):
    """
    prefix + target을 한 번에 모델에 넣어서
    target 전체 시퀀스("Tim Cook")에 대한 확률을 계산.
    """
    # 1) prefix와 target을 붙여서 토크나이징
    #    (모델이 prefix 다음에 target을 생성한다고 가정)
    combined_text = prefix + target
    combined_inputs = tokenizer(combined_text, return_tensors='pt')
    
    # GPU 사용한다면 .to('cuda') 추가
    combined_inputs = {k: v.to(model.device) for k, v in combined_inputs.items()}

    if block_opt == 1:
        block_tokenized_b = tokenizer(' '+block_text, return_tensors='pt')
        block_tokenized_nb = tokenizer(block_text, return_tensors='pt')
        
        length_b = len(block_tokenized_b['input_ids'][0])-1 # eliminate <begin of text>
        start_idx_b = block_tokenized_b['input_ids'][0][1]
        
        
        length_nb = len(block_tokenized_nb['input_ids'][0])-1 # eliminate <begin of text>
        start_idx_nb = block_tokenized_nb['input_ids'][0][1]

        attention_idx = 0
        attention_mask = combined_inputs['attention_mask']
        for i in range(0, len(combined_inputs['input_ids'][0])):
            if combined_inputs['input_ids'][0][i] == start_idx_b:
                attention_mask[:, attention_idx:attention_idx+length_b] = 0
                break
            elif combined_inputs['input_ids'][0][i] == start_idx_nb:
                attention_mask[:, attention_idx:attention_idx+length_nb] = 0
        combined_inputs['attention_mask'] = attention_mask

    # 2) 모델 Forward
    with torch.no_grad():
        outputs = model(**combined_inputs)
    # outputs.logits.shape: [batch_size=1, seq_len, vocab_size]
    
    # 3) prefix 구간과 target 구간의 토큰 인덱스 파악
    prefix_inputs = tokenizer(prefix, return_tensors='pt')
    prefix_length = prefix_inputs['input_ids'].shape[1]  # prefix의 토큰 개수
    
    # 전체 인풋 토큰
    combined_ids = combined_inputs['input_ids'][0]  # (seq_len,) 형태
    # target에 해당하는 토큰은 prefix_length 이후부터 끝까지
    target_ids = combined_ids[prefix_length:]
    
    # 4) target 시퀀스 각 토큰 확률 구하기
    #    logits[i-1]이 i번째 토큰의 확률 분포를 의미함.
    #    따라서 i = prefix_length 부터는 prefix_length-1이 target의 첫번째 토큰을 예측하는 logit이 됨.
    log_probs = []
    for i in range(prefix_length, len(combined_ids)):
        # i번째 토큰을 예측한 logits은 outputs.logits[0, i-1]
        # (i가 prefix_length일 때가 target의 첫 번째 토큰이 됨)
        if i == 0:
            # 시퀀스의 첫 번째 토큰(i=0)은 이전 토큰이 없으므로 스킵
            continue
        
        token_id = combined_ids[i]
        # logits에서 해당 위치 인덱스에 대한 softmax -> 그 중 token_id의 확률
        logits_i = outputs.logits[0, i-1]
        prob_i = F.softmax(logits_i, dim=-1)[token_id]
        
        # 로그 확률로 바꿔서 append
        log_probs.append(torch.log(prob_i))
    
    # 5) 모든 토큰의 로그 확률을 합산 -> exp해서 최종 확률
    total_log_prob = torch.sum(torch.stack(log_probs))
    total_prob = torch.exp(total_log_prob).item()
    
    return total_prob

'''
새로 만든 데이터셋, subject hop sentence + fact knowledge
'''
def hop_sentence_prob(model, tokenizer, data, check_col, block_opt, opt = 0):
    count = 0
    for i in tqdm(range(len(data))):
        prefix_text = data[i]['generated_sentences'][check_col]['sentence_with_hop_word'][0] + data[i]['prompt'].format(data[i]['subject'])
        block_text = data[i][check_col][0]
        
        if opt == 0:
            target_text = ' '+data[i]['fact_knowledge']
            t = 'fact_knowledge'
        else:
            target_text = ' ' + data[i]['edited_knowledge']
            t = 'edited_knowledge'

        prob = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text, block_text, block_opt =block_opt)
        count += prob
    logging.info(f'{check_col} with {t} : {round(count / len(data), 6)}')
    return (count / len(data))

def make_chat_temp(hop_sentence, question):
    char = [
        {"role": "user", "content": hop_sentence},
    ]
    text = tokenizer.apply_chat_template(char, tokenize = False) + '<|start_header_id|>assistant<|end_header_id|>\n\n' + question
    text = text.replace("<|begin_of_text|>", "", 1)
    #text = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + hop_sentence + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + question
    return text

def hop_sentence_prob_chat(model, tokenizer, data, check_col, block_opt, opt = 0):
    count = 0
    for i in tqdm(range(len(data))):
        prefix_text = make_chat_temp(data[i]['generated_sentences'][check_col]["sentence_with_hop_word"][0], data[i]['prompt'].format(data[i]['subject']))
        block_text = data[i][check_col][0]
        if opt == 0:
            target_text = ' '+data[i]['fact_knowledge']
            t = 'fact_knowledge'
        else:
            target_text = ' ' + data[i]['edited_knowledge']
            t = 'edited_knowledge'
        prob = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text, block_text, block_opt =block_opt)
        count += prob
    logging.info(f'{check_col} with {t} : {round(count / len(data), 6)}')
    return (count / len(data))


if __name__ == "__main__":
        

    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from utils.setting import set_seed, now

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default = "meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--log_dir', type=str, default = "../../.cache")
    parser.add_argument('--cache_dir', type=str, default = "../../.cache")
    parser.add_argument('--data_dir', type=str, default = "../../data/final_5000.json")
    parser.add_argument('--cuda_num', type=str, default = 0)
    parser.add_argument('--block_opt', type=int, default = 0)
    parser.add_argument('--chat_opt', type=int, default = 0)

    args = parser.parse_args()
    model_name = args.model_name
    log_dir = args.log_dir
    cache_dir = args.cache_dir
    data = args.data_dir
    cuda_num = args.cuda_num
    block_opt = args.block_opt
    chat_opt = args.chat_opt
    
    # 로깅 설정: 'app.log' 파일에 INFO 수준 이상의 로그를 저장
    logging.basicConfig(
        filename=log_dir,                     # 로그 파일 이름 지정
        filemode='a',                           # 파일 모드 ('a'는 append, 'w'는 overwrite)
        level=logging.INFO,                     # 로그 수준 설정
        format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 형식 지정
    )

    
    logging.info(f'model_name : {model_name}')
    logging.info(f'data : {data}')
    if block_opt == 1:
        logging.info(f'Attention blocking = True')
    else:
        logging.info('Attention blocking = False')
        
    if chat_opt == 1:
        logging.info('Chat Template = True')
    else:
        logging.info('Chat Template = False')
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

    now()

    seed = 42
    set_seed(seed)


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir = cache_dir, torch_dtype=torch.float16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'

    #model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = "../.cache", torch_dtype=torch.bfloat16).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, torch_dtype=torch.float16).to('cuda')



    with open(data, 'r') as file:
        data = json.load(file) 
        

    hop_cols = ['sbj_hop_test', 'obj_true_hop_test', 'obj_new_hop_test']
    for check_col in hop_cols:
        for opt in range(0, 2):
            if chat_opt==1:
                logging.info(hop_sentence_prob_chat(model, tokenizer, data, check_col, block_opt = block_opt, opt = opt))
            else:
                logging.info(hop_sentence_prob(model, tokenizer, data, check_col, block_opt = block_opt, opt = opt))