from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import pandas as pd
import random
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from setting import set_seed
set_seed(42)
# k : output data_size(train + test)
k=10
df_base = pd.read_excel("../../data/one_hop/experiment_1_df.xlsx")
new_columns = ['sbj_hop_train', 'sbj_hop_test', 'obj_true_hop_train', 'obj_true_hop_test', 'obj_new_hop_train', 'obj_new_hop_test']
for col in new_columns:
    df_base[col] = ''

with open('../../counterfact_memit.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]
    
def extract(df, data):
    result = []
    for idx in range(len(df)):
        dt = data[df.loc[idx, 'index']]
        
        result.append({
            "case_id": str(df.loc[idx, "index"]),
            "prompt": dt['requested_rewrite']['prompt'],
            "subject" : df.loc[idx, 'subject'],
            "fact_knowledge" : df.loc[idx, 'obj_true'],
            "edited_knowledge" : df.loc[idx, 'obj_new'],
            "relation_id": dt['requested_rewrite']['relation_id'],
            "rephrased_prompt" : dt['paraphrase_prompts'][0],
            "locality_prompt" : dt['neighborhood_prompts'][0],
            "locality_ground_truth" : df.loc[idx, 'obj_true'],
            
            
            "sbj_hop_train" : df.loc[idx, 'sbj_hop_train'].split(','),
            "sbj_hop_test" : df.loc[idx, 'sbj_hop_test'].split(','),
            "obj_true_hop_train" : df.loc[idx, 'obj_true_hop_train'].split(','),
            "obj_true_hop_test" : df.loc[idx, 'obj_true_hop_test'].split(','),
            "obj_new_hop_train" : df.loc[idx, 'obj_new_hop_train'].split(','),
            "obj_new_hop_test" : df.loc[idx, 'obj_new_hop_test'].split(','),
        })
    return result

# BERT 모델과 토크나이저 로드
model_name = 'bert-base-uncased'  # 또는 다른 BERT 모델
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def word_sort(word, hop, count = 10):
    hop_list = hop.split(',')
    word_tok = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
    data = dict()
    with torch.no_grad():
        word_emb = model(**word_tok).last_hidden_state[:, 0, :]
        for hop in hop_list:
            hop_tok = tokenizer(hop, return_tensors='pt', padding=True, truncation=True)
            hop_emb = model(**hop_tok).last_hidden_state[:, 0, :]
            data[hop] = F.cosine_similarity(word_emb, hop_emb).item()
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    result = []
    for wd in sorted_data:
        result.append(wd)
    word_lists = result[:count]
    return ','.join(word_lists)

# experiment a
df = df_base.copy()

def filter_a(word_list, k=10):
    word_lists = word_list.split(',')
    selected_list = random.sample(word_lists, k)
    return ','.join(selected_list[0::2]), ','.join(selected_list[1::2])

for i in range(len(df)):
    df.loc[i, 'sbj_hop_train'], df.loc[i, 'sbj_hop_test'] = filter_a(df.loc[i, 'sbj_one_hop'], k)
    df.loc[i, 'obj_true_hop_train'], df.loc[i, 'obj_true_hop_test'] = filter_a(df.loc[i, 'obj_one_hop'], k)
    df.loc[i, 'obj_new_hop_train'], df.loc[i, 'obj_new_hop_test'] = filter_a(df.loc[i, 'obj_new_one_hop'], k)

df_a = extract(df, data)     

with open("df_exp1_a.json", "w") as json_file:
    json.dump(df_a, json_file, indent=4, ensure_ascii=False)
    
# experiment b
df = df_base.copy()

def filter_b(word, hop, k = 10):
    ans = word_sort(word, hop, k).split(',')
    return ','.join(ans[0::2]), ','.join(ans[1::2])

for i in tqdm(range(len(df))):
    df.loc[i, 'sbj_hop_train'], df.loc[i, 'sbj_hop_test'] = filter_b(df.loc[i, 'subject'],df.loc[i, 'sbj_one_hop'], k)   
    df.loc[i, 'obj_true_hop_train'], df.loc[i, 'obj_true_hop_test'] = filter_b(df.loc[i, 'obj_true'], df.loc[i, 'obj_one_hop'], k)
    df.loc[i, 'obj_new_hop_train'], df.loc[i, 'obj_new_hop_test'] = filter_b(df.loc[i, 'obj_new'], df.loc[i, 'obj_new_one_hop'], k)

df_b = extract(df, data)     

with open("df_exp1_b.json", "w") as json_file:
    json.dump(df_b, json_file, indent=4, ensure_ascii=False)

# experiment c
df = df_base.copy()

def filter_c(word_list, k = 10):
    word_lists = word_list.split(',')
    # 저장할 때 내림차순 정렬 해놨음
    reversed_list = word_lists[::-1]
    word_lists = reversed_list[:k]
    return ','.join(word_lists[0::2]), ','.join(word_lists[1::2])

for i in range(len(df)):
    df.loc[i, 'sbj_hop_train'], df.loc[i, 'sbj_hop_test'] = filter_c(df.loc[i, 'sbj_one_hop'], k)
    df.loc[i, 'obj_true_hop_train'], df.loc[i, 'obj_true_hop_test'] = filter_c(df.loc[i, 'obj_one_hop'], k)
    df.loc[i, 'obj_new_hop_train'], df.loc[i, 'obj_new_hop_test'] = filter_c(df.loc[i, 'obj_new_one_hop'], k)
    
df_c = extract(df, data)     

with open("df_exp1_c.json", "w") as json_file:
    json.dump(df_c, json_file, indent=4, ensure_ascii=False)

# experiment d
df = df_base.copy()

def filter_d(word, hop, k = 10):
    reversed_list = hop.split(',')[::-1]
    ans = word_sort(word, hop, k*2).split(',')
    ans_sorted = sorted(ans, key = lambda x : reversed_list.index(x))
    ans_sorted = ans_sorted[:k]
    return ','.join(ans_sorted[0::2]), ','.join(ans_sorted[1::2])

for i in tqdm(range(len(df))):
    df.loc[i, 'sbj_hop_train'], df.loc[i, 'sbj_hop_test'] = filter_d(df.loc[i, 'subject'],df.loc[i, 'sbj_one_hop'], k)   
    df.loc[i, 'obj_true_hop_train'], df.loc[i, 'obj_true_hop_test'] = filter_d(df.loc[i, 'obj_true'], df.loc[i, 'obj_one_hop'], k)
    df.loc[i, 'obj_new_hop_train'], df.loc[i, 'obj_new_hop_test'] = filter_d(df.loc[i, 'obj_new'], df.loc[i, 'obj_new_one_hop'], k)
df_d = extract(df, data)     

with open("df_exp1_d.json", "w") as json_file:
    json.dump(df_d, json_file, indent=4, ensure_ascii=False)

# experiment e
df = df_base.copy()

def filter_e(word, hop, k = 10):
    reversed_list = hop.split(',')[::-1]
    word_lists = reversed_list[:k*2]
    
    ans = word_sort(word, ','.join(word_lists), k).split(',')
    return ','.join(ans[0::2]), ','.join(ans[1::2])

for i in tqdm(range(len(df))):
    df.loc[i, 'sbj_hop_train'], df.loc[i, 'sbj_hop_test'] = filter_e(df.loc[i, 'subject'],df.loc[i, 'sbj_one_hop'], k)   
    df.loc[i, 'obj_true_hop_train'], df.loc[i, 'obj_true_hop_test'] = filter_e(df.loc[i, 'obj_true'], df.loc[i, 'obj_one_hop'], k)
    df.loc[i, 'obj_new_hop_train'], df.loc[i, 'obj_new_hop_test'] = filter_e(df.loc[i, 'obj_new'], df.loc[i, 'obj_new_one_hop'], k)

df_e = extract(df, data)     

with open("df_exp1_e.json", "w") as json_file:
    json.dump(df_e, json_file, indent=4, ensure_ascii=False)
    