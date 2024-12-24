import pandas as pd
import requests
import time
from tqdm import tqdm


df = pd.read_excel('df_full_obj_new.xlsx')

df['obj_new_one_hop'] = df['obj_id']



def get_wikidata_labels(entity_id):
    # SPARQL 쿼리 엔드포인트
    url = "https://query.wikidata.org/sparql"

    # SPARQL 쿼리 (입력받은 entity_id에 해당하는 값들의 영어 레이블을 가져옴)
    query = f"""
    SELECT DISTINCT ?valueLabel WHERE {{
      wd:{entity_id} ?relation ?value.
      VALUES ?relation {{
        wdt:P138 wdt:P641 wdt:P37 wdt:P27 wdt:P495 wdt:P19 wdt:P413 wdt:P101
        wdt:P264 wdt:P740 wdt:P39 wdt:P159 wdt:P449 wdt:P364 wdt:P30 wdt:P463
        wdt:P1412 wdt:P140 wdt:P127 wdt:P131 wdt:P108 wdt:P136 wdt:P36 wdt:P176
        wdt:P20 wdt:P276 wdt:P190 wdt:P937 wdt:P103 wdt:P106 wdt:P1303 wdt:P407
        wdt:P178 wdt:P17
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    
    """

    # 요청 헤더
    headers = {
        "Accept": "application/sparql-results+json"
    }

    # 요청 전송
    response = requests.get(url, headers=headers, params={'query': query})

    # 결과를 JSON으로 변환
    data = response.json()

    # 영어 레이블 반환
    if data['results']['bindings']:
        return [result['valueLabel']['value'] for result in data['results']['bindings']]
    else:
        return []

def main(search_string):

    entity_id = search_string
    if entity_id:
        labels = get_wikidata_labels(entity_id)
        # print(f"The English labels for entity ID {entity_id} are: {labels}")
    else:
        print(f"No entity found for search string '{search_string}'.")
    return labels

    

# 예시 사용


for i in tqdm(range(len(df))):
    try:
        ans = main(df.iloc[i]['obj_id'])
        time.sleep(2)
        df.loc[i, 'obj_new_one_hop'] = ','.join(ans)
        
    except:
        print(f'error occured in {i}')
        df.loc[i, 'obj_new_one_hop'] = 'null'
        time.sleep(2)
        

df.to_excel('df_one_hop_obj_new_full.xlsx', index = False)