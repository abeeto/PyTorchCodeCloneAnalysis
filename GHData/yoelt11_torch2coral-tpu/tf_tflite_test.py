import tensorflow as tf
import numpy as np
import torch

if __name__ == '__main__':

    # load tflite
    tflite_interpreter =  tf.lite.Interpreter('tflite-model/tflite_model.tflite')
    tflite_interpreter.allocate_tensors()
    input_index = tflite_interpreter.get_input_details()[0]["index"]
    output_index = tflite_interpreter.get_output_details()[0]["index"]

    # load tensorflow 

    tf_model = tf.saved_model.load('tensorflow-model/tf_model')
    inference_fn = tf_model.signatures["serving_default"]
    # dummy input 
    B, H, W, C = 1, 64, 256, 3
    X = np.full((B, H, W, C), 1.0, dtype=np.float32)

    # run inference tflite
    tflite_interpreter.set_tensor(input_index, X.astype('int8'))
    tflite_interpreter.invoke()

    # run inference tf
    tf_output = inference_fn(tf.constant(X))

    # get outputs
    tflite_outputs = tflite_interpreter.get_tensor(output_index)
    
    # test outputs
    print(f"tf output: {torch.tensor(tf_output['output'].numpy()).topk(3).indices}")
    print(f'tflite output: {torch.tensor(tflite_outputs).topk(3).indices}')
