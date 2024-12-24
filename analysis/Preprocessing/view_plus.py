import requests
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_wikipedia_views(qid, years=3):
    headers = {
        "User-Agent": "MyApp/1.0 (https://mywebsite.com; myemail@example.com)"
    }

    # Wikidata API 요청 URL 및 제목 확인
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(entity_url, headers=headers)
    
    if response.status_code != 200:
        return -1  # 요청 실패 시 -1 반환

    try:
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

        # 제목 인코딩 (URL에서 사용 가능한 형태로)
        from urllib.parse import quote
        title_encoded = quote(title, safe='')

        # Wikimedia API 요청 URL 설정
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title_encoded}/monthly/{start_str}/{end_str}"
        views_response = requests.get(url, headers=headers)
        
        if views_response.status_code != 200:
            return -1  # 요청 실패 시 -1 반환

        # 전체 조회수 합산 후 반환
        views_data = views_response.json().get('items', [])
        total_views = sum(item['views'] for item in views_data) if views_data else -1
        
        return total_views
    except:
        return -1  # 예외 발생 시 -1 반환

# 데이터 읽기
df = pd.read_excel("one_hop/raw_df.xlsx")

# 'view' 열이 없으면 생성하고 -1로 초기화
if 'view' not in df.columns:
    df['view'] = -1  # -1은 조회 실패 또는 아직 조회하지 않은 상태를 나타냄

max_iterations = 5  # 최대 반복 횟수 설정
iteration = 0

while True:
    iteration += 1
    print(f"\n===== Iteration {iteration} =====")

    # 'view' 값이 -1인 행의 인덱스 목록 추출
    indices_to_process = df[df['view'] == -1].index.tolist()
    #print(f"남은 처리 대상: {len(indices_to_process)}개")

    if not indices_to_process:
        print("모든 QID의 조회수를 성공적으로 가져왔습니다.")
        break  # 모든 조회수를 가져왔으므로 루프 종료

    if iteration > max_iterations:
        print("최대 반복 횟수에 도달했습니다.")
        break  # 최대 반복 횟수 초과 시 루프 종료

    # 조회수 가져오기
    for idx in tqdm(indices_to_process):
        qid = df.at[idx, 'subject_id']
        total_views = get_wikipedia_views(qid)
        df.at[idx, 'view'] = total_views  # 결과 업데이트

        if total_views == -1:
            print(f"QID {qid}의 조회수를 가져오지 못했습니다.")
        #else:
            #print(f"QID {qid}: {total_views} 조회수")

    # 중간 결과 저장 (옵션)
    #df.to_excel("view_intermediate.xlsx", index=False)
    #print("중간 결과를 'view_intermediate.xlsx' 파일로 저장했습니다.")

# 최종 결과 저장
df.to_excel("view_final.xlsx", index=False)
print("최종 결과를 'view_final.xlsx' 파일로 저장했습니다.")
