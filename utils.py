import requests
import yaml
import json

def load_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("JSON 데이터 도착") 
        return response.json()
    else:
        print("실패") 
        return None

def load_jsonl_from_path(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config