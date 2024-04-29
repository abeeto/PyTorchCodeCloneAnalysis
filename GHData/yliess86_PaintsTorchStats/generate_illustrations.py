import requests
import base64
import os

from io import BytesIO
from tqdm import tqdm
from PIL import Image

def img2b64(img):
    buff = BytesIO()
    img.save(buff, format='PNG')
    b64  = b'data:image/png;base64,' + base64.b64encode(buff.getvalue())
    b64  = str(b64)[2:-1]
    return b64
    
def b642img(b64):
    string = b64.replace('data:image/png;base64,', '')
    img    = Image.open(BytesIO(base64.b64decode(string)))
    return img

url                = 'https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer'
models             = ['PaperRS', 'CustomSS', 'CustomSD']

base_path          = './data/StudyDataset'
linearts_path      = os.path.join(base_path, 'LineArt')
hints_path         = os.path.join(base_path, 'Hint')
illustrations_path = os.path.join(base_path, 'Illustration')

lineart_files      = [os.path.join(linearts_path, f) for f in os.listdir(linearts_path)]
hint_files         = [os.path.join(hints_path, f) for f in os.listdir(hints_path)]

if not os.path.exists(illustrations_path):
    os.mkdir(illustrations_path)

for i, files in enumerate(tqdm(zip(sorted(lineart_files), sorted(hint_files)), total=len(hint_files))):
    lineart_f, hint_f = files
    lineart           = Image.open(lineart_f)
    hint              = Image.open(hint_f)
    
    for model in models:
        data = {
            'sketch'  : img2b64(lineart),
            'hint'    : img2b64(hint),
            'opacity' : 1.0,
            'model'   : model
        }
        
        resp = requests.post(url, json=data)
        data = resp.json()
        
        if 'colored' in data:
            colored      = data['colored']
            illustration = b642img(colored)
        
            illustration.save(os.path.join(
                illustrations_path, 
                f"{i:03d}_{model}.png"
            ))
        