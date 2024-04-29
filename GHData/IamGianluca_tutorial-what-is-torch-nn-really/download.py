from pathlib import Path

import requests

from constants import DATA_PATH, PATH, FILENAME


PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
