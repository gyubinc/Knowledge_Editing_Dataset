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

##################################################################
# 1. 중복되는 코어 함수(토큰화 → generate → 결과문자열 검사)는 그대로 둡니다.
##################################################################
def compute_sequence_blocking_probability(model, tokenizer, prefix, target1, target2, block_text, block_opt=0):
    """
    prefix + target을 한 번에 모델에 넣어서
    target1, target2가 결과문자열에 등장하는지 확인해
    (1,1), (1,0), (0,1), (0,0)을 반환.
    """
    combined_text = prefix
    
    combined_inputs = tokenizer(
        combined_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings
    )

    # attention_mask와 pad_token_id 추가
    attention_mask = combined_inputs.get('attention_mask', None)
    combined_inputs['pad_token_id'] = tokenizer.pad_token_id
    combined_inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in combined_inputs.items()
    }

    block_text = block_text.strip()
    
    # block_opt==1인 경우 attention_mask에서 block_text 토큰화 부분을 0으로 지정
    if block_opt == 1:
        block_tokenized_b = tokenizer(' ' + block_text, return_tensors='pt')
        block_tokenized_nb = tokenizer(block_text, return_tensors='pt')
        
        length_b = len(block_tokenized_b['input_ids'][0]) - 1  # <bos> 제거
        start_idx_b = block_tokenized_b['input_ids'][0][1]
        
        length_nb = len(block_tokenized_nb['input_ids'][0]) - 1
        start_idx_nb = block_tokenized_nb['input_ids'][0][1]

        attention_idx = 0
        attention_mask = combined_inputs['attention_mask']
        for i in range(len(combined_inputs['input_ids'][0])):
            if combined_inputs['input_ids'][0][i] == start_idx_b:
                attention_mask[:, attention_idx:attention_idx+length_b] = 0
                break
            elif combined_inputs['input_ids'][0][i] == start_idx_nb:
                attention_mask[:, attention_idx:attention_idx+length_nb] = 0
                break
            # else:
            #     print("error occured")
        combined_inputs['attention_mask'] = attention_mask

    with torch.no_grad():
        generated_output = model.generate(
            input_ids=combined_inputs['input_ids'],
            attention_mask=combined_inputs['attention_mask'],
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id
        )

    # 생성된 토큰만 추출
    input_length = combined_inputs['input_ids'].shape[1]
    generated_tokens = generated_output[0][input_length:]
    
    # 생성된 토큰을 디코딩
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # target1, target2 각각이 생성 문자열(decoded_output)에 포함되는지 확인
    found_t1 = int(target1.lower() in decoded_output.lower())
    found_t2 = int(target2.lower() in decoded_output.lower())
    return found_t1, found_t2

##################################################################
# 2. 단어 치환 함수는 그대로 사용
##################################################################
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

    flags = re.IGNORECASE if case_insensitive else 0

    if allow_partial_match:
        pattern_str = re.escape(target_word)
    else:
        # 단어 경계 매칭
        pattern_str = rf'(?<!\w){re.escape(target_word)}(?!\w)'

    pattern = re.compile(pattern_str, flags)

    if not pattern.search(text) and raise_when_not_found:
        print("실패")
        print(text)
        print(target_word)

    replaced_text = pattern.sub(replacement_word, text)
    return replaced_text

##################################################################
# 3. Chat 템플릿 생성 함수는 그대로 사용
##################################################################
def make_chat_temp(hop_sentence, question, tokenizer):
    """
    chat_opt가 활성화된 경우 사용할 prefix 텍스트
    """
    char = [
        {"role": "user", "content": hop_sentence},
    ]
    text = tokenizer.apply_chat_template(char, tokenize=False) + '<|start_header_id|>assistant<|end_header_id|>\n\n' + question
    text = text.replace("<|begin_of_text|>", "", 1)
    return text

##################################################################
# 4. **단일 함수**로 중복 로직을 통합
##################################################################
def calculate_hop_sentence_prob(
    model, tokenizer, data, check_col,
    block_opt=0,
    chat_opt=False,
    random_opt=False,
    random_word_opt=False
):
    """
    - hop_sentence_prob / hop_sentence_prob_chat / hop_sentence_prob_random 등
      기존 여러 함수를 통합한 함수.
    - chat_opt=True면 챗 템플릿(make_chat_temp) 사용
    - random_opt=True면 i+1 인덱스를 사용 (랜덤 인덱스 대신 예시상 'i+1'이었으므로 그대로)
    - random_word_opt=True면 prefix_text에서 block_text를 교체
    """
    count_true = 0
    count_new = 0
    
    n_data = len(data)//2
    for i in tqdm(range(n_data)):
        # 1) random_opt에 따라 prefix/target을 가져올 인덱스 k 결정
        k = i + n_data

        # 2) prefix_text 생성
        #    chat_opt에 따라 make_chat_temp 사용여부 결정
        if chat_opt:
            # chat인 경우: 
            #   - random_opt=False면 i 사용, True면 k 사용(원본 코드 차이 있음)
            #   - 기존 hop_sentence_prob_random_chat에서 prefix_text = make_chat_temp(data[k]...)
            #     → random 시 k, 일반 시 i
            #   - 여기서는 random_opt에 맞춰 k를 쓰도록 통일
            if random_opt:
                prefix_text = make_chat_temp(
                    data[k]['generated_sentences'][check_col]["sentence_with_hop_word"][0],
                    data[i]['prompt'].format(data[i]['subject']),
                    tokenizer
                )
            else:
                prefix_text = make_chat_temp(
                    data[i]['generated_sentences'][check_col]["sentence_with_hop_word"][0],
                    data[i]['prompt'].format(data[i]['subject']),
                    tokenizer
                )
                
        else:
            if random_opt:
                prefix_text = (
                    data[k]['generated_sentences'][check_col]['sentence_with_hop_word'][0]
                    + data[i]['prompt'].format(data[i]['subject'])
                )
            # chat 아닐 때(일반 텍스트)
            else:
                prefix_text = (
                    data[i]['generated_sentences'][check_col]['sentence_with_hop_word'][0]
                    + data[i]['prompt'].format(data[i]['subject'])
                )

        prefix_text = data[i]['prompt'].format(data[i]['subject'])
        
        # 3) block_text 결정
        #    random_opt → block_text 역시 k로 가져오는 경우가 있음
        #    (기존 로직 유지: hop_sentence_prob_random 시 block_text = data[k][check_col][0])
        #    일반은 i
        #    random_word_opt는 추가로 prefix_text 내용을 치환
        if random_opt:
            block_text = data[k][check_col][0]
        else:
            block_text = data[i][check_col][0]

        # 4) random_word_opt=True이면 prefix_text에서 block_text를 치환
        #    실제 치환 대상 단어는 "i의 block_text"이고, 치환값은 "k의 block_text" — (원본 코드 로직)
        #    chat/비챗 가리지 않고 동일 처리
        if random_word_opt:
            # 실제로는 hop_sentence_prob_random_word()에서
            # prefix_text = data[i][generated_sentences...] + data[i]['prompt'] (i)
            # block_text = data[i][check_col][0] (i)
            # 치환 대상: data[k][check_col][0]
            # 여기서는 random_opt와 별개로, 'i' vs 'k'를 의도대로 맞춰야 하므로
            #   - 랜덤이든 아니든, random_word_opt 자체가 'i→k 치환' 기능
            #   - 코드에서 보듯, block_text는 i가 맞음
            original_block_text = data[i][check_col][0].strip()
            replacement_word = data[k][check_col][0].strip()
            try:
                prefix_text = replace_word_in_text(prefix_text, original_block_text, replacement_word)
            except:
                logging.info(f"error occurred while replace_word_in_text")
                logging.info(prefix_text)
                logging.info(original_block_text)

        # 5) fact_knowledge / edited_knowledge
        target_text1 = data[i]['fact_knowledge']
        target_text2 = data[i]['edited_knowledge']

        # 6) 생성결과 확률(포함 여부) 계산
        prob_true, prob_new = compute_sequence_blocking_probability(
            model, tokenizer,
            prefix_text,
            target_text1,
            target_text2,
            block_text,
            block_opt=block_opt
        )
        count_true += prob_true
        count_new += prob_new

    x1 = round(count_true / n_data, 6)
    x2 = round(count_new / n_data, 6)
    
    t1 = 'fact_knowledge'
    t2 = 'edited_knowledge'
    logging.info(f'{check_col} with {t1} : {count_true} / {n_data} = {x1} ({count_true/n_data}) ')
    logging.info(f'{check_col} with {t2} : {count_new} / {n_data} {x2} ({count_new/n_data}) ')
    return x1, x2


##################################################################
# 5. 메인 실행부 예시
##################################################################
if __name__ == "__main__":
    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from utils.setting import set_seed, now

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--log_dir', type=str, default="experiment.log")
    parser.add_argument('--cache_dir', type=str, default="../../.cache")
    parser.add_argument('--data_dir', type=str, default='df_exp1_e_5000_with_generated_sentences.json')
    parser.add_argument('--cuda_num', type=str, default='0')
    parser.add_argument('--block_opt', type=int, default=0)
    parser.add_argument('--chat_opt', type=int, default=0)
    parser.add_argument('--random_opt', type=int, default=0)
    parser.add_argument('--random_word', type=int, default=0)

    args = parser.parse_args()
    model_name = args.model_name
    cache_dir = args.cache_dir
    data_path = args.data_dir
    cuda_num = args.cuda_num
    block_opt = args.block_opt
    log_dir = args.log_dir
    chat_opt = bool(args.chat_opt)
    random_opt = bool(args.random_opt)
    random_word_opt = bool(args.random_word)
    
    # 로깅 설정
    logging.basicConfig(
        filename=log_dir,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("*" * 50)
    logging.info(f'model_name : {model_name}')
    logging.info(f'data : {data_path}')
    logging.info(f'Attention blocking = {bool(block_opt)}')
    logging.info(f'Chat Template = {chat_opt}')
    logging.info(f'Random = {random_opt}')
    logging.info(f'Random_word = {random_word_opt}')

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    now()

    seed = 42
    set_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).to('cuda')

    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)

    # 이 예시에서는 hop_cols라는 3개 컬럼에 대해 순회
    hop_cols = ['sbj_hop_test', 'obj_true_hop_test', 'obj_new_hop_test']


    # 본격적으로 확률(생성 결과 포함 여부) 계산
    for check_col in hop_cols:
        x1, x2 = calculate_hop_sentence_prob(
            model, tokenizer, data,
            check_col=check_col,
            block_opt=block_opt,
            chat_opt=chat_opt,
            random_opt=random_opt,
            random_word_opt=random_word_opt
        )
        logging.info(f"Result => {check_col}: fact={x1}, edited={x2}")
