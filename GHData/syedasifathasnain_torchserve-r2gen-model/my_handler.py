
from PIL import Image
import zipfile

with zipfile.ZipFile('serve.zip', 'r') as zip_ref:
    zip_ref.extractall()
from handler import MyHandler
_service = MyHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    # image1 = Image.open('frontal.png').convert('RGB')
    # image2 = Image.open('lateral.png').convert('RGB')

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
