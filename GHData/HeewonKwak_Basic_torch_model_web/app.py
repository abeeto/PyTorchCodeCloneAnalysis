import io
import json
# import cv2
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
# from flask import Flask, jsonify, request, send_file


app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
      file = request.files['file']
      img_bytes = file.read()
      class_id, class_name = get_prediction(image_bytes=img_bytes)
      return jsonify({'class_id': class_id, 'class_name': class_name})

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         img_bytes = file.read()
#         # # class_id, class_name = get_prediction(image_bytes=img_bytes)
#         # image = transform_image(image_bytes=img_bytes)
#         # # return jsonify({'image': image})
#         # return send_file(image, mimetype='image/jpg')
#         img = cv2.imread(img_bytes, cv2.IMREAD_COLOR)
#         img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
#         # img_bytes = open(abs_path, 'rb').read() 속도 차이는 미미함
#         return jsonify({'image': img_bytes})


if __name__ == '__main__':
    app.run()