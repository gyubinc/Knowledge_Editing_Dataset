import os
import sys
import json
import torch
import logging
import argparse
import re
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    """텍스트 내 target_word를 replacement_word로 교체한다."""
    if not target_word:
        raise ValueError("대상 단어(target_word)가 비어 있습니다.")

    flags = re.IGNORECASE if case_insensitive else 0

    if allow_partial_match:
        # 부분 일치: target_word가 어떤 위치에든 등장하면 매칭
        pattern_str = re.escape(target_word)
    else:
        # 정확한 단어 경계 매칭
        pattern_str = rf'(?<!\w){re.escape(target_word)}(?!\w)'

    pattern = re.compile(pattern_str, flags)

    if not pattern.search(text) and raise_when_not_found:
        # 에러를 발생시키고 싶을 때만
        print("실패")
        print(text)
        print(target_word)

    replaced_text = pattern.sub(replacement_word, text)
    return replaced_text


def compute_sequence_blocking_probability(model, tokenizer, prefix, target, block_text, block_opt=0):
    """
    prefix + target을 한 번에 모델에 넣어서
    target 전체 시퀀스에 대한 확률을 계산.
    """
    # 1) 토크나이징
    combined_text = prefix + target
    combined_inputs = tokenizer(combined_text, return_tensors='pt')
    combined_inputs = {k: v.to(model.device) for k, v in combined_inputs.items()}

    # Attention Blocking 옵션 처리
    if block_opt == 1:
        block_tokenized_b = tokenizer(' ' + block_text, return_tensors='pt')
        block_tokenized_nb = tokenizer(block_text, return_tensors='pt')

        length_b = len(block_tokenized_b['input_ids'][0]) - 1  # 첫 번째 토큰(시작 토큰) 제거 길이
        start_idx_b = block_tokenized_b['input_ids'][0][1]
        length_nb = len(block_tokenized_nb['input_ids'][0]) - 1
        start_idx_nb = block_tokenized_nb['input_ids'][0][1]

        attention_mask = combined_inputs['attention_mask']
        attention_idx = 0
        for i in range(len(combined_inputs['input_ids'][0])):
            if combined_inputs['input_ids'][0][i] == start_idx_b:
                attention_mask[:, attention_idx:attention_idx+length_b] = 0
                break
            elif combined_inputs['input_ids'][0][i] == start_idx_nb:
                attention_mask[:, attention_idx:attention_idx+length_nb] = 0
        combined_inputs['attention_mask'] = attention_mask

    # 2) 모델 Forward
    with torch.no_grad():
        outputs = model(**combined_inputs)

    # 3) prefix와 target 토큰 구간 파악
    prefix_inputs = tokenizer(prefix, return_tensors='pt')
    prefix_length = prefix_inputs['input_ids'].shape[1]  # prefix의 토큰 개수

    combined_ids = combined_inputs['input_ids'][0]
    # target 구간 = prefix_length 이후부터 끝까지
    target_ids = combined_ids[prefix_length:]

    # 4) target 시퀀스 각 토큰 확률 계산
    log_probs = []
    for i in range(prefix_length, len(combined_ids)):
        if i == 0:
            continue
        token_id = combined_ids[i]
        logits_i = outputs.logits[0, i-1]
        prob_i = F.softmax(logits_i, dim=-1)[token_id]
        log_probs.append(torch.log(prob_i))

    # 5) 모든 토큰의 로그 확률을 합산 후 exp
    total_log_prob = torch.sum(torch.stack(log_probs))
    total_prob = torch.exp(total_log_prob).item()
    return total_prob


def make_chat_temp(tokenizer, hop_sentence, question):
    """Chat 템플릿 형태로 prefix를 구성."""
    char = [
        {"role": "user", "content": hop_sentence},
    ]
    text = (
        tokenizer.apply_chat_template(char, tokenize=False)
        + '<|start_header_id|>assistant<|end_header_id|>\n\n'
        + question
    )
    # (원본에서 <|begin_of_text|> 제거)
    text = text.replace("<|begin_of_text|>", "", 1)
    return text


def hop_sentence_prob_all(
    model,
    tokenizer,
    data,
    check_col: str,
    block_opt: int = 0,
    opt: int = 0,
    chat_opt: int = 0,
    random_opt: int = 0,
    random_word: int = 0
):
    """
    중복 로직을 하나로 모은 함수.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    data : list
        JSON 로드된 데이터 목록
    check_col : str
        'sbj_hop_test', 'obj_true_hop_test', 'obj_new_hop_test' 중 하나
    block_opt : int
        Attention blocking 여부 (0 or 1)
    opt : int
        0이면 fact_knowledge, 1이면 edited_knowledge
    chat_opt : int
        0이면 일반, 1이면 Chat 템플릿
    random_opt : int
        0이면 자기 자신, 1이면 i+1(랜덤) 사용
    random_word : int
        0이면 hop 단어 그대로, 1이면 다른 문서의 hop 단어로 교체

    Returns
    -------
    float
        평균 확률
    """
    # target 텍스트(fact vs edited) 결정
    target_col = 'fact_knowledge' if opt == 0 else 'edited_knowledge'

    count = 0
    size = len(data)//2
    for i in tqdm(range(size)):
        # --- 1) 인덱스 결정 (랜덤/자기 자신) ---
        #    i+1번째 문서를 사용할 때, 마지막 원소면 맨 앞으로 돌아감
        if random_opt == 1:
            idx_for_prefix = i + size
        else:
            idx_for_prefix = i

        # --- 2) prefix 텍스트 구성 ---
        hop_sentence_str = data[idx_for_prefix]['generated_sentences'][check_col]['sentence_with_hop_word'][0]
        prompt_str = data[i]['prompt'].format(data[i]['subject'])

        if chat_opt == 1:
            prefix_text = make_chat_temp(tokenizer, hop_sentence_str, prompt_str)
        else:
            prefix_text = hop_sentence_str + prompt_str
        prefix_text = prompt_str
        # block_text는 prefix를 구성할 때 사용한 hop 문서(= idx_for_prefix)
        block_text = data[idx_for_prefix][check_col][0]

        # --- 3) hop 단어 교체 여부 (random_word) ---
        # random_word == 1이면, "현재 문서(i)의 hop 단어"를 "idx_for_prefix 문서의 hop 단어"로 바꿔치기하는 로직
        if random_word == 1:
            # 현재 문서(i)의 hop 단어
            hop_word_i = data[i][check_col][0]
            # prefix_text 내 hop_word_i를 block_text로 교체
            try:
                prefix_text = replace_word_in_text(prefix_text, hop_word_i, block_text)
            except Exception as e:
                logging.info("error occured")
                logging.info(prefix_text)
                logging.info(hop_word_i)

        # --- 4) target 텍스트 준비 ---
        target_text = ' ' + data[i][target_col].strip()

        # --- 5) 확률 계산 ---
        prob = compute_sequence_blocking_probability(
            model, tokenizer, prefix_text, target_text, block_text, block_opt=block_opt
        )
        count += prob
    mean_prob = count / size
    logging.info(f'{check_col} with {target_col} : {round(mean_prob, 3)}')
    return mean_prob


def main():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from utils.setting import set_seed, now  # 내부 유틸 (사용자 정의 모듈)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--log_dir', type=str, default="../../.cache")
    parser.add_argument('--cache_dir', type=str, default="../../.cache")
    parser.add_argument('--data_dir', type=str, default="../../data/final_5000.json")
    parser.add_argument('--cuda_num', type=str, default='0')
    parser.add_argument('--block_opt', type=int, default=0)
    parser.add_argument('--chat_opt', type=int, default=0)
    parser.add_argument('--random_opt', type=int, default=0)
    parser.add_argument('--random_word', type=int, default=0)

    args = parser.parse_args()
    model_name = args.model_name
    log_dir = args.log_dir
    cache_dir = args.cache_dir
    data_path = args.data_dir
    cuda_num = args.cuda_num
    block_opt = args.block_opt
    chat_opt = args.chat_opt
    random_opt = args.random_opt
    random_word = args.random_word

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

    if block_opt:
        logging.info('Attention blocking = True')
    else:
        logging.info('Attention blocking = False')

    if chat_opt:
        logging.info('Chat Template = True')
    else:
        logging.info('Chat Template = False')

    if random_opt:
        logging.info("Random = True")
    else:
        logging.info("Random = False")

    if random_word:
        logging.info("Random word = True")
    else:
        logging.info("Random word = False")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

    now()  # time stamp
    set_seed(42)  # 재현성

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).to('cuda')

    # 데이터 로드
    with open(data_path, 'r') as f:
        data = json.load(f)

    # hop_cols 반복
    hop_cols = ['sbj_hop_test', 'obj_true_hop_test', 'obj_new_hop_test']
    for check_col in hop_cols:
        # opt: 0 => fact_knowledge, 1 => edited_knowledge
        for opt in [0, 1]:
            val = hop_sentence_prob_all(
                model=model,
                tokenizer=tokenizer,
                data=data,
                check_col=check_col,
                block_opt=block_opt,
                opt=opt,
                chat_opt=chat_opt,
                random_opt=random_opt,
                random_word=random_word
            )
            logging.info(val)


if __name__ == "__main__":
    main()
