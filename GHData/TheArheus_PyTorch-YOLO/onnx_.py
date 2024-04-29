import warnings
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import onnx2keras
import numpy as np

warnings.filterwarnings('ignore')
#sess = ort.InferenceSession('model.onnx')

model = onnx.load_model("model.onnx")
k_model = onnx2keras.onnx_to_keras(onnx_model=model, input_names=["input_data"], change_ordering=True, verbose=False)
print(k_model.layers)