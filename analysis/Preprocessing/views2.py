import requests
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib.parse

def get_wikipedia_views(qid, years=3):
    headers = {
        "User-Agent": "MyApp/1.0 (https://mywebsite.com; myemail@example.com)"
    }

    # Fetch entity data from Wikidata
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(entity_url, headers=headers)
    if response.status_code != 200:
        return -1
    
    # Extract the Wikipedia title
    try:
        entity_data = response.json()
        title = entity_data['entities'][qid].get('sitelinks', {}).get('enwiki', {}).get('title', None)
    except Exception as e:
        print(f"Error parsing JSON for QID {qid}: {e}")
        return -1

    if not title:
        return -1

    # URL-encode the title
    title_encoded = urllib.parse.quote(title, safe='')

    # Set up date range for pageviews
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years*365)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    # Fetch pageviews from Wikimedia API
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title_encoded}/monthly/{start_str}/{end_str}"
    views_response = requests.get(url, headers=headers)
    if views_response.status_code != 200:
        return -1

    # Sum up the total views
    try:
        views_data = views_response.json().get('items', [])
        total_views = sum(item['views'] for item in views_data) if views_data else -1
    except Exception as e:
        print(f"Error parsing views data for title {title}: {e}")
        return -1
        
    return total_views


# Read the Excel file
df = pd.read_excel("view_final2.xlsx")
print(df.head())

error_count = 0
subject_ids = df['subject_id'].tolist()
updated_views = []

for idx in tqdm(range(len(subject_ids))):
    if df.loc[idx, 'view'] == -1:
        try:
            total_views = get_wikipedia_views(subject_ids[idx])
            print(f"Subject ID: {subject_ids[idx]}, Total Views: {total_views}")
            if total_views == -1:
                print('Error fetching views')
                error_count += 1
            updated_views.append(total_views)
        except Exception as e:
            print(f'Error for subject_id {subject_ids[idx]}: {e}')
            updated_views.append(-1)
    else:
        # Preserve the existing 'view' value
        updated_views.append(df.loc[idx, 'view'])

df['view'] = updated_views

print(f"Number of errors: {error_count}")

# Save the updated DataFrame back to the Excel file
df.to_excel("view_final2.xlsx", index=False)
