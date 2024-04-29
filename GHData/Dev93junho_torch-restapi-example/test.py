import requests
import json

res = requests.post('http://localhost:5000/predict',
                    files ={"file" : open('./_static/img/sample.png', 'rb')})

result = res.json()

with open("result.txt", "w") as f:
    json.dump(result, f)