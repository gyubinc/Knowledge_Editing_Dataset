import random
import numpy as np
import torch
import transformers
import pandas as pd

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def now():
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)

now()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # GPU seed 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # PyTorch 재현성 설정 (CUDNN)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
    


# 시드를 고정할 값 설정
seed = 42
set_seed(seed)

import pandas as pd
df = pd.read_excel("preprocessed_df2.xlsx")
df.head()
# 조건 정의
condition = (df['sbj_hop_num'] == 0) | (df['obj_true_hop_num'] == 0) | (df['obj_new_hop_num'] == 0)
# 조건을 만족하는 행 제거
df_filtered = df[~condition]

df = df_filtered.reset_index(drop = True)


from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = "../.cache")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir = "../.cache").to('cuda')


from tqdm import tqdm

model.eval()

for i in tqdm(range(len(df))):
    subject = df.loc[i, 'subject']
    sbj_hops = df.loc[i, 'sbj_one_hop']
    x = dict()
    for word in sbj_hops.split(','):
        
        # 확률 계산할 문장
        sentence = f"{subject} and {word}"
        # 토큰화
        inputs = tokenizer(sentence, return_tensors="pt").to('cuda')
        # 모델 출력 (logits)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # 소프트맥스: 확률 계산
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # 각 토큰의 로그 확률 계산
        input_ids = inputs['input_ids']
        log_probs = torch.log(probs[0, torch.arange(input_ids.size(-1)), input_ids[0]])
        # 전체 문장의 로그 확률 계산
        total_log_prob = log_probs.sum().item()
        # 결과 출력
        x[word] = total_log_prob
        #print(f" '{sentence}': {total_log_prob}")
    sorted_keys = sorted(x, key=x.get, reverse=True)
    df.loc[i, 'sbj_one_hop'] = ','.join(sorted_keys)

df.to_excel("ex_f_pre2.xlsx", index = False)