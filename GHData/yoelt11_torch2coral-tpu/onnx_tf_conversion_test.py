import tensorflow as tf
import onnxruntime as ort
import torch
import numpy as np

if __name__ == '__main__':
    
    # create dummy input
    B, H, W, C = 1, 64, 256, 3
    X = np.full((B, H, W, C), 1.0, dtype=np.float32)
   
    # load tf model
    tf_model = tf.saved_model.load('tensorflow-model/tf_model')
    inference_fn = tf_model.signatures["serving_default"]

    # load onnx model
    onnx_model = ort.InferenceSession("onnx-model/onnx_model.onnx")
   
    # run inference tf
    tf_output = inference_fn(tf.constant(X))

    # run inference onnx
    onnx_output = onnx_model.run(None, {'input': X})[0]
    
    # debug outputs
    print(f"tf output: {torch.tensor(tf_output['output'].numpy()).topk(3).indices}")
    print(f'onnx output: {torch.tensor(onnx_output[0,:]).topk(3).indices}')

