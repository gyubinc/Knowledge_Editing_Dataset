import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

'''
해당 함수에서는 BERT를 통해 모든 데이터를 BERT 기준 내림차순으로 정렬해서
OUTPUT에 ['sbj_hop', 'obj_true_hop', 'obj_new_hop'] COLUMN에 담아서 내보냄ㄴ
INPUT : preprocessed_df2 ( 원시 데이터에서 전처리만 거친 데이터 )
OUTPUT : BERT_preprocessed_df2 ( BERT 기준 유사도 정렬을 마친 데이터 )
'''

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델과 토크나이저 로드
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# 데이터 로드
df = pd.read_excel('../../data/one_hop/preprocessed_df2.xlsx')
df.fillna('null', inplace = True)

# word_sort 함수: 토큰화, 임베딩 계산 및 유사도 정렬
def word_sort(word, hop_list, count=10):
    # 입력 단어와 hop_list를 토큰화하고 GPU로 전송
    word_tok = tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(device)
    hop_toks = tokenizer(hop_list, return_tensors='pt', padding=True, truncation=True).to(device)

    with torch.no_grad():
        # 입력 단어와 hop_list의 임베딩 계산
        word_emb = model(**word_tok).last_hidden_state[:, 0, :]
        hop_embs = model(**hop_toks).last_hidden_state[:, 0, :]
        # cosine similarity 계산
        similarities = F.cosine_similarity(word_emb, hop_embs)
    
    # 유사도 순으로 정렬
    sorted_indices = similarities.argsort(descending=True)
    sorted_data = [hop_list[i] for i in sorted_indices.cpu().numpy()]
    return sorted_data

# 데이터프레임 행별 처리 함수
def process_row(row):
    result = {}
    if row['sbj_hop_num'] != 0:
        result['sbj_hop'] = ','.join(word_sort(row['subject'], row['sbj_one_hop'].split(','), count=10))
    else:
        result['sbj_hop'] = 'null'

    if row['obj_true_hop_num'] != 0:
        result['obj_true_hop'] = ','.join(word_sort(row['subject'], row['obj_one_hop'].split(','), count=10))
    else:
        result['obj_true_hop'] = 'null'

    if row['obj_new_hop_num'] != 0:
        result['obj_new_hop'] = ','.join(word_sort(row['obj_new'], row['obj_new_one_hop'].split(','), count=10))
    else:
        result['obj_new_hop'] = 'null'

    return pd.Series(result)

# tqdm으로 진행률 표시
tqdm.pandas()

# 데이터 처리
df_sorted = df.copy()
df_sorted[['sbj_hop', 'obj_true_hop', 'obj_new_hop']] = df.progress_apply(process_row, axis=1)

# 결과 저장
df_sorted.to_excel("BERT_preprocessed_df2.xlsx", index=False)
