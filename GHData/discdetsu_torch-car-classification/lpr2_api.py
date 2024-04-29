import requests

def get_license_plate(frame):
    url = "https://api.aiforthai.in.th/lpr-v2"
    payload = {'crop': '1', 'rotate': '1'}
    files = {'image':open(frame, 'rb')}
 
    headers = {
        'Apikey': "zizPXb2o5x74uZQTR0XGlDj6ev89hG4p",
        }
    
    
    response = requests.post( url, files=files, data = payload, headers=headers)
    try: 
        return response.json()[0]['lpr']
    except:
        return 'Not found'