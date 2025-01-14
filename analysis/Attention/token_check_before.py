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
import re

def compute_sequence_blocking_probability(model, tokenizer, prefix, target1, target2, block_text, block_opt = 0):
    """
    prefix + target을 한 번에 모델에 넣어서
    target 전체 시퀀스("Tim Cook")에 대한 확률을 계산.
    """
    
    combined_text = prefix
    # 수정된 코드
    combined_inputs = tokenizer(combined_text, return_tensors='pt', padding=True, truncation=True, max_length=model.config.max_position_embeddings)

    # attention_mask와 pad_token_id 추가
    attention_mask = combined_inputs.get('attention_mask', None)
    combined_inputs['pad_token_id'] = tokenizer.pad_token_id
    combined_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in combined_inputs.items()}

    block_text = block_text.strip()
    if block_opt == 1:
        # b = word with pre-blank space
        # nb = word with out pre-blank space
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
            else:
                print("error occured")
        combined_inputs['attention_mask'] = attention_mask

    with torch.no_grad():
        generated_output = model.generate(input_ids=combined_inputs['input_ids'], 
                                          attention_mask=combined_inputs['attention_mask'],
                                          max_new_tokens=50,  # 생성할 새로운 토큰 수
                                          pad_token_id=tokenizer.pad_token_id)

    # 생성된 토큰만 추출
    input_length = combined_inputs['input_ids'].shape[1]
    generated_tokens = generated_output[0][input_length:]
    
    # 생성된 토큰을 디코딩
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if target1.lower() in decoded_output.lower():
        if target2.lower() in decoded_output.lower():
            # both generated (target true, target new)
            return 1, 1
        else:
            # target true only
            return 1, 0
    elif target2.lower() in decoded_output.lower():
        # target new only
        return 0, 1
    else:
        # Nothing
        return 0, 0
    
'''
새로 만든 데이터셋, subject hop sentence + fact knowledge
'''
def hop_sentence_prob(model, tokenizer, data, check_col, block_opt, opt = 0):
    count = 0
    for i in tqdm(range(len(data))):
        prefix_text = data[i]['generated_sentences'][check_col]['sentence_with_hop_word'][0] + data[i]['prompt'].format(data[i]['subject'])
        block_text = data[i][check_col][0]
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2

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
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2


def hop_sentence_prob_random(model, tokenizer, data, check_col, block_opt, opt = 0):
    count = 0
    for i in tqdm(range(len(data))):
        if i == (len(data)-1):
            k = 0
        else:
            k = i+1
        prefix_text = data[k]['generated_sentences'][check_col]['sentence_with_hop_word'][0] + data[i]['prompt'].format(data[i]['subject'])
        block_text = data[k][check_col][0]
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2

def hop_sentence_prob_random_word(model, tokenizer, data, check_col, block_opt, opt = 0):
    count_true = 0
    count_new = 0
    for i in tqdm(range(len(data))):
        if i == (len(data)-1):
            k = 0
        else:
            k = i+1
        prefix_text = data[i]['generated_sentences'][check_col]['sentence_with_hop_word'][0] + data[i]['prompt'].format(data[i]['subject'])
        block_text = data[i][check_col][0]
        try:
            prefix_text = replace_word_in_text(prefix_text, block_text, data[k][check_col][0])
        except:
            logging.info(f"error occured")
            logging.info(prefix_text)
            logging.info(block_text)
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2


def hop_sentence_prob_random_chat(model, tokenizer, data, check_col, block_opt, opt = 0):

    for i in tqdm(range(len(data))):
        if i == (len(data)-1):
            k = 0
        else:
            k = i+1
        prefix_text = make_chat_temp(data[k]['generated_sentences'][check_col]["sentence_with_hop_word"][0], data[i]['prompt'].format(data[i]['subject']))
        block_text = data[k][check_col][0]
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2

def hop_sentence_prob_random_chat_word(model, tokenizer, data, check_col, block_opt, opt = 0):
    count_true = 0
    count_new = 0
    for i in tqdm(range(len(data))):
        if i == (len(data)-1):
            k = 0
        else:
            k = i+1
        prefix_text = make_chat_temp(data[i]['generated_sentences'][check_col]["sentence_with_hop_word"][0], data[i]['prompt'].format(data[i]['subject']))
        block_text = data[i][check_col][0].strip()
        try:
            prefix_text = replace_word_in_text(prefix_text, block_text, data[k][check_col][0])
        except:
            logging.info(f"error occured")
            logging.info(prefix_text)
            logging.info(block_text)
        target_text1 = data[i]['fact_knowledge']
        t1 = 'fact_knowledge'
        target_text2 = data[i]['edited_knowledge']
        t2 = 'edited_knowledge'
        prob_true, prob_new = compute_sequence_blocking_probability(model, tokenizer, prefix_text, target_text1, target_text2, block_text, block_opt =block_opt)
        count_true += prob_true
        count_new += prob_new
    x1 = round(count_true / len(data), 6)
    x2 = round(count_new / len(data), 6)
    logging.info(f'{check_col} with {t1} : {x1}')
    logging.info(f'{check_col} with {t2} : {x2}')
    return x1, x2




class WordNotFoundError(Exception):
    """대상 단어가 텍스트 내에 존재하지 않을 때 발생하는 예외."""
    def __init__(self, target_word, message=None):
        if message is None:
            message = f"텍스트 내에 대상 단어 '{target_word}'가(이) 존재하지 않습니다."
        super().__init__(message)
        self.target_word = target_word

def replace_word_in_text(
    text: str,
    target_word: str,
    replacement_word: str,
    case_insensitive: bool = True,
    allow_partial_match: bool = True,
    raise_when_not_found: bool = False
) -> str:

    if not target_word:
        raise ValueError("대상 단어(target_word)가 비어 있습니다.")

    # 정규식 플래그 설정
    flags = re.IGNORECASE if case_insensitive else 0

    if allow_partial_match:
        # 부분 일치: 단순히 target_word가 어떤 위치에든 등장하면 매칭
        #   예) "human" -> "humane"에서도 "human" 부분 찾아 교체
        pattern_str = re.escape(target_word)
    else:
        # 정확한 단어 경계 매칭 (원본과 유사)
        #   s/es 등의 간단한 복수형 정도만 허용하고 싶다면 추가 조정
        #   아래에서는 기본만 예시
        pattern_str = rf'(?<!\w){re.escape(target_word)}(?!\w)'

    pattern = re.compile(pattern_str, flags)

    # 매칭되는 부분이 있는지 확인
    if not pattern.search(text) and raise_when_not_found:
        # 원하는 경우에만 에러 발생
        print("실패")
        print(text)
        print(target_word)

    # 교체 수행
    replaced_text = pattern.sub(replacement_word, text)
    return replaced_text





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
    parser.add_argument('--block_opt', type=int, default = 0)
    parser.add_argument('--chat_opt', type=int, default = 0)
    parser.add_argument('--random_opt', type=int, default=0)
    parser.add_argument('--random_word', type=int, default=0)

    args = parser.parse_args()
    model_name = args.model_name
    cache_dir = args.cache_dir
    data = args.data_dir
    cuda_num = args.cuda_num
    block_opt = args.block_opt
    log_dir = args.log_dir
    chat_opt = args.chat_opt
    random_opt = args.random_opt
    random_word = args.random_word
    
    # 로깅 설정: 'app.log' 파일에 INFO 수준 이상의 로그를 저장
    logging.basicConfig(
        filename=log_dir,                     # 로그 파일 이름 지정
        filemode='a',                           # 파일 모드 ('a'는 append, 'w'는 overwrite)
        level=logging.INFO,                     # 로그 수준 설정
        format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 형식 지정
    )
    
    logging.info(f'model_name : {model_name}')
    logging.info(f'data : {data}')
    
    if block_opt:
        logging.info(f'Attention blocking = True')
    else:
        logging.info('Attention blocking = False')
        
    if chat_opt:
        logging.info('Chat Template = True')
    else:
        logging.info('Chat Template = False')
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

    if random_opt:
        logging.info("Random = True")
    else:
        logging.info("Random = False")
        
    if random_word:
        logging.info("Random_word = True")
    else:
        logging.info("Random_word = False")
    
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
        for i in range(len(data)):
            if i == (len(data)-1):
                k = 0
            else:
                k = i+1
            prefix_text = make_chat_temp(data[i]['generated_sentences'][check_col]["sentence_with_hop_word"][0], data[i]['prompt'].format(data[i]['subject']))
            block_text = data[i][check_col][0].strip()
            replace_word_in_text(prefix_text, block_text, data[k][check_col][0].strip())


    hop_cols = ['sbj_hop_test', 'obj_true_hop_test', 'obj_new_hop_test']
    for check_col in hop_cols:
        if random_opt:
            if random_word:
                logging.info("good")
                if chat_opt:
                    logging.info(hop_sentence_prob_random_chat_word(model, tokenizer, data, check_col, block_opt = block_opt))
                else:
                    logging.info(hop_sentence_prob_random_word(model, tokenizer, data, check_col, block_opt = block_opt))
                
            else:
                logging.info("bad")
                if chat_opt:
                    logging.info(hop_sentence_prob_random_chat(model, tokenizer, data, check_col, block_opt = block_opt))
                else:
                    logging.info(hop_sentence_prob_random(model, tokenizer, data, check_col, block_opt = block_opt))
            
        else:
            logging.info("bad")
            if chat_opt:
                logging.info(hop_sentence_prob_chat(model, tokenizer, data, check_col, block_opt = block_opt))
            else:
                logging.info(hop_sentence_prob(model, tokenizer, data, check_col, block_opt = block_opt))