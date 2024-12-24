import pandas as pd
from tqdm import tqdm
import json
import logging

# 파일에 로그 저장
logging.basicConfig(
    filename='freq2.log', 
    filemode='a',       
    level=logging.INFO, 
    format='%(message)s' 
)

'''
해당 함수에서는 FREQUENCY에 따라 단어를 정렬함.
OUTPUT은 기존 COLUMN을 그대로 유지한 채로 기존 COLUMN 내부만 정렬함
INPUT : BERT_preprocessed_df2 ( BERT 기준 유사도 정렬을 마친 데이터 )
OUTPUT : BERT_FREQ_preprocessed_df2 ( BERT, FREQ 정렬 모두 마친 데이터 )
-> 해당 데이터를 기준으로 이제 뽑기만 하면 됨

BERT 유사도 정렬을 먼저 하는 이유 = freq로 자를 때 min_count로 자를 경우 데이터 개수가 줄어들기 때문에
'''
# 데이터 로드
df = pd.read_excel('../../data/one_hop/BERT_preprocessed_df2.xlsx')
df.fillna('null', inplace = True)
dx = df.copy()

with open('../../data/one_hop/word_frequency2.json', "r") as f:
    word_counts = json.load(f)

def filtering(word, word_counts, min_count=2):
    if word == 'null':
        return 0, 'null'
    word_list = word.split(',')
    word_list = [word for word in word_list if word_counts.get(word, 0) >= min_count]
    word_list = sorted(word_list, key=lambda x: word_counts.get(x, 0), reverse=True)
    num = len(word_list)
    return num, ','.join(word_list)

def filtering2(dx, min_count, k):
    df = dx.copy()
    for i in range(len(df)):
        df.loc[i, 'sbj_hop_num'], df.loc[i, 'sbj_one_hop'] = filtering(df.loc[i, 'sbj_one_hop'], word_counts, min_count)
        df.loc[i, 'obj_true_hop_num'], df.loc[i, 'obj_one_hop'] = filtering(df.loc[i, 'obj_one_hop'], word_counts, min_count)
        df.loc[i, 'obj_new_hop_num'], df.loc[i, 'obj_new_one_hop'] = filtering(df.loc[i, 'obj_new_one_hop'], word_counts, min_count)

    well_list = []
    second_list = []
    for i in range(len(df)):
        if (df.loc[i, 'sbj_hop_num']>=k and df.loc[i, 'obj_true_hop_num']>=k and df.loc[i, 'obj_new_hop_num']>=k):
            well_list.append(i)
    logging.info(f'Total number of data with (freq >= {min_count}) and (every length >= {k}) is {len(well_list)}({round(100 * len(well_list) / 21919, 2)}%)')
    
    for i in range(len(df)):
        if (df.loc[i, 'sbj_hop_num']>=k or df.loc[i, 'obj_true_hop_num']>=k or df.loc[i, 'obj_new_hop_num']>=k):
            second_list.append(i)
    logging.info(f'Total number of data with (freq >= {min_count}) and (one of them length >= {k}) is {len(second_list)}({round(100 * len(second_list) / 21919, 2)}%)')
    return df


min_count = 2
k = 0
df = filtering2(dx, min_count, k)

# for min_count in tqdm(range(0, 6)):
#     for k in range(0, 6):
#         df = filtering2(dx, min_count, k)


# 결과 저장
df.to_excel("BERT_FREQ2_preprocessed_df2.xlsx", index=False)
