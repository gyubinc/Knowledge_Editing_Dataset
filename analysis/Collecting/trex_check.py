import requests
import time
import pandas as pd
from tqdm import tqdm


def get_qid_exact_label(text):
    url = 'https://query.wikidata.org/sparql'
    query = '''
    SELECT ?item WHERE {
      ?item rdfs:label "%s"@en .
    }
    ''' % text
    headers = {
        'Accept': 'application/sparql-results+json'
    }
    params = {
        'query': query,
        'format': 'json'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception("SPARQL 쿼리에 실패했습니다: 상태 코드 %d" % response.status_code)
    data = response.json()
    results = data['results']['bindings']
    if len(results) == 1:
        qid_url = results[0]['item']['value']
        qid = qid_url.split('/')[-1]
        return qid
    else:
        return None

# 사용 예시
text = "Seoul"
qid = get_qid_exact_label(text)
print(qid == None)  # 출력 예시: Q937

df2 = pd.read_excel("no_trex_df_5452_2024113.xlsx")
df3 = df2[['index', 'subject', 'subject_id']]
df3.loc[:, 'subject_new_id'] = 'null'
df3 = df3.reset_index(drop = True)


for i in tqdm(range(len(df3))):
    try:
        text = df3['subject'][i]
        qid = get_qid_exact_label(text)
        if (qid == None):
            pass
        elif (qid != None):
            df3.loc[i, 'subject_new_id'] = qid
        time.sleep(1)
    except:
        pass
    
    
df3.to_excel("no_trex_after_check_5452_20241113.xlsx", index = False)