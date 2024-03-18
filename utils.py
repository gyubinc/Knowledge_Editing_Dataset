import requests
import yaml

def load_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("JSON 데이터 도착") 
        return response.json()
    else:
        print("실패") 
        return None
    
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config