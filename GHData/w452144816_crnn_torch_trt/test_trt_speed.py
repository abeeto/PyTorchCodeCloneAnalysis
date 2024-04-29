import os
import numpy as np
import tensorrt as trt
import trt_utils
import time
import alphabets_english as alp
import dataset
from PIL import Image
TRT_LOGGER = trt.Logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_path = './demo/img_14.jpg'
engine_path = './new_crnn_english.trt'
batch = 10
width = 1024
alphabets = alp.alphabet


batch_images = np.zeros(shape=(32, 1, 32, width), dtype=np.float32)
image = Image.open(img_path).convert('L')
image = image.resize([int(width/1.5),32])
image = np.array(image, np.float32)
image = image / 255
for i in range(batch):
    batch_images[i, 0, :, :int(width/1.5)] = image


def decode(resd):
    _, w = resd.shape
    texts = []
    for i in range(batch):
        l = resd[i]
        char_list=[]
        for i in range(w):
            if l[i] != 0 and (not (i > 0 and l[i - 1] == l[i])):
                char_list.append(alphabets[l[i] - 1])
        texts.append(''.join(char_list))
    return texts

with trt_utils.get_engine(
        engine_path, TRT_LOGGER) as engine, engine.create_execution_context() as context:

    inputs, outputs, bindings, stream = trt_utils.allocate_buffers(engine)
    # batch_images = np.random.random_sample([32, 1, 32, 1024])
    for i in range(10):
        start = time.time()
        inputs[0].host = np.array(batch_images, dtype=np.float32,
                                  order='C')
        trt_outputs = trt_utils.do_inference_v2(context, bindings=bindings,
                                             inputs=inputs, outputs=outputs,
                                             stream=stream)
        trt_outputs = trt_outputs[0].reshape([int(width/4+1), 32, len(alphabets)+1])
        trt_outputs = np.argmax(trt_outputs, axis=2)
        # print(len(batch_outputs))
        trt_outputs = np.transpose(trt_outputs,(1,0))[:batch, :]
        print(decode(trt_outputs))
        end = time.time()
        print(end - start)