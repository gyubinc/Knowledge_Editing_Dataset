import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import math
from tqdm import tqdm

df = pd.read_excel('../../data/df_241110.xlsx')

def get_labels(qids, batch_size=100):
    labels = {}
    total_batches = math.ceil(len(qids) / batch_size)
    for i in tqdm(range(total_batches)):
        batch_qids = qids[i*batch_size:(i+1)*batch_size]
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        qid_values = " ".join("wd:%s" % qid for qid in batch_qids)
        query = """
        SELECT ?qid ?label WHERE {
          VALUES ?qid { %s }
          ?qid rdfs:label ?label .
          FILTER (lang(?label) = "en")
        }
        """ % qid_values
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                qid = result["qid"]["value"].split('/')[-1]
                label = result["label"]["value"]
                labels[qid] = label
        except Exception as e:
            print(f"Error querying batch {i+1}/{total_batches}: {e}")
    return labels

# QID 리스트 가져오기
qids = df['subject_id'].tolist()

# 레이블 딕셔너리 가져오기
labels = get_labels(qids, batch_size=50)  # 배치 크기를 50으로 설정

# 레이블을 데이터프레임에 추가
df['wiki_subject'] = df['subject_id'].map(labels)


cut = 0
for i in tqdm(range(len(df))):
    if df['subject'][i] != df['wiki_subject'][i]:
        cut += 1
print(cut)


df.to_excel("exact_qid_match.xlsx", index = False)

