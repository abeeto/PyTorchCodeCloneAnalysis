from flask import Flask, jsonify, request
import io
import torchvision.transforms as transforms 
from PIL import Image

from torchvision import models
import json


app = Flask(__name__)
imagenet_class_index = json.load(open('./_static/imagenet_class_index.json'))
model = models.densenet121(pretrained=True) # used pretrain model
model.eval() # 'eval'mode --> only use to inference  

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

# with open('./_static/img/sample.png', 'rb') as f:
#     image_bytes = f.read()
#     tensor = transform_image(image_bytes=image_bytes)
#     print(tensor)
    
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    pred_idx = str(y_hat.item())
    return imagenet_class_index[pred_idx]

# with open('./_static/img/sample.png', 'rb') as f:
#     image_bytes = f.read()
#     print(get_prediction(image_bytes=image_bytes))

@app.route('/')
def hello():
    return "access flask app"
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id' : class_id, 'class_name' : class_name})
    
    
if __name__ == '__main__':
    app.debug = True
    app.run()

