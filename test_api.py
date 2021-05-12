import requests
import json
response = requests.get('http://127.0.0.1:8000/predict/มีอะไร')
print(json.loads(response.content))
