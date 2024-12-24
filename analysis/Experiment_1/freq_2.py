import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
import json
import random
from setting import set_seed

set_seed(42)

df = pd.read_excel("../../data/one_hop/preprocessed_df.xlsx")
df.fillna('null',inplace=True)

with open('../../data/one_hop/word_frequency.json', "r") as f:
    word_counts = json.load(f)

min_count = 2
def filtering(word, word_counts, min_count=2):
    if word == 'null':
        return 0, 'null'
    word_list = word.split(',')
    word_list = [word for word in word_list if word_counts.get(word, 0) >= min_count]
    word_list = sorted(word_list, key=lambda x: word_counts.get(x, 0), reverse=True)
    num = len(word_list)
    return num, ','.join(word_list)

for i in range(len(df)):
    df.loc[i, 'sbj_hop_num'], df.loc[i, 'sbj_one_hop'] = filtering(df.loc[i, 'sbj_one_hop'], word_counts, min_count)
    df.loc[i, 'obj_true_hop_num'], df.loc[i, 'obj_one_hop'] = filtering(df.loc[i, 'obj_one_hop'], word_counts, min_count)
    df.loc[i, 'obj_new_hop_num'], df.loc[i, 'obj_new_one_hop'] = filtering(df.loc[i, 'obj_new_one_hop'], word_counts, min_count)
    
well_list = []
k=20
for i in range(len(df)):
    if (df.loc[i, 'sbj_hop_num']>=k and df.loc[i, 'obj_true_hop_num']>=k and df.loc[i, 'obj_new_hop_num']>=k):
        well_list.append(i)
print(f'Total number of data with (freq > {min_count}) and (length > {k}) is {len(well_list)}')

sample_list = random.sample(well_list, 1000)
sample_list.sort()
df = df.iloc[sample_list]

df = df.reset_index(drop = True)

df.to_excel("experiment_1_df.xlsx", index = False)
