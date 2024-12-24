import requests
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm  # 변경된 부분

def get_wikipedia_views(qid, years=3):
    headers = {
        "User-Agent": "MyApp/1.0 (https://mywebsite.com; myemail@example.com)"
    }

    # Wikidata API 요청 URL 및 제목 확인
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(entity_url, headers=headers)
    
    # 문서 제목 추출
    entity_data = response.json()
    title = entity_data['entities'][qid].get('sitelinks', {}).get('enwiki', {}).get('title', None)
    if not title:
        return -1

    # 조회수를 가져오기 위한 Wikimedia API 호출 설정
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years*365)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    # Wikimedia API 요청 URL 설정
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/monthly/{start_str}/{end_str}"
    views_response = requests.get(url, headers=headers)

    # 전체 조회수 합산 후 반환
    views_data = views_response.json().get('items', [])
    total_views = sum(item['views'] for item in views_data) if views_data else -1
    
    return total_views


df = pd.read_excel("one_hop/one_hop_sbj.xlsx")
df.head()

a = 0
sbj_list = df[['index', 'subject_id']].copy()  # 원본 DataFrame 보호
sb = df['subject_id']
sbj_list['view'] = np.zeros(len(df))

end_list = []

for qid in tqdm(sb):
    try:
        total_views = get_wikipedia_views(qid)
        
        #print(total_views)
        if total_views == -1:
            print('error')
            a += 1
        end_list.append(total_views)
    except:
        print('error2')
        end_list.append(-1)
sbj_list['view'] = end_list

print(a)

sbj_list.to_excel("view.xlsx", index = False)
