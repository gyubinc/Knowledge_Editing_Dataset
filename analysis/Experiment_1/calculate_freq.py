from collections import Counter
import json
import pandas as pd

df = pd.read_excel("../../data/one_hop/preprocessed_df2.xlsx")
df.fillna('null',inplace=True)

freq_list = []
for i in range(len(df)):
    freq_list += df.loc[i, 'sbj_one_hop'].split(',')
    freq_list += df.loc[i, 'obj_one_hop'].split(',')
    freq_list += df.loc[i, 'obj_new_one_hop'].split(',')

# 1. Counter를 사용해 빈도수 계산
word_counts = Counter(freq_list)

# 2. value(빈도수)의 내림차순으로 정렬
sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

# 3. JSON 파일로 저장
with open("word_frequency2.json", "w") as f:
    json.dump(sorted_word_counts, f, indent=4)