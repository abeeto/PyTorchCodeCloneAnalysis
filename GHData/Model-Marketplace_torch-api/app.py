from flask import Flask, request, jsonify
import torch
import numpy as np
from model import FFNN

app = Flask(__name__)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

model = load_checkpoint("news_model.pth")

@app.route('/')
def index():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    x = data["data"]
    x = torch.from_numpy(np.array(x)).to(torch.float32)
    output = model(x)
    _, prediction = torch.max(output, 1)
    prediction = prediction.item()
    return jsonify({ "prediction": prediction })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
