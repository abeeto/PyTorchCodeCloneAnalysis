import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import cv2
import glob
from skimage.color import gray2rgb
#https://zhuanlan.zhihu.com/p/387853124
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


trt_logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(trt_logger)

# def load_engine(trt_logger):
#     # engine_file_path = '/home/timtran/Desktop/TensorRT/char_adlt.engine'
#     engine_file_path = '/home/timtran/Desktop/TensorRT/resnet18-mnist.engine'
#     with open(engine_file_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
#         return runtime.deserialize_cuda_engine(f.read())


# ===========================================================================
# ===========================================================================
# engine = load_engine(trt_logger)
# context = engine.create_execution_context()

class HostDeviceMem(object):
    #Simple helper data class that's a little nicer to use than a 2-tuple.

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def alloc_buf_N(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []

    stream = cuda.Stream()

    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding)) * -1 * engine.max_batch_size
        # size = trt.volume(engine.get_binding_shape(binding)) * 1
        # size = 1*1*28*28
        # size = 49152 = 16*3072 for outputs

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # dtype = # <class 'numpy.float32'> for both input and output

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # host_mem = [0. 0. 0. ... 0. 0. 0.], 
        # host_mem.shape) = (1572864,) and (49152,) for inputs and outputs respectively

        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            # print("inputs alloc_buf_N", inputs)
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            # print("outputs alloc_buf_N", outputs)

    return inputs, outputs, bindings, stream
 
def do_inference_v2(engine, context, inputs, bindings, outputs, stream):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Return only the host outputs.
    return [out.host for out in outputs]

# with open('/home/timtran/Desktop/TensorRT/resnet18-mnist.engine', "rb") as f:
#     engine = runtime.deserialize_cuda_engine(f.read())

# with engine.create_execution_context() as context:
#     batch_size = 128
#     img = np.ones([128,1,28,28])
#     context.active_optimization_profile = 0
#     context.set_binding_shape(0, (batch_size, 1, 28, 28))
#     inputs, outputs, bindings, stream = alloc_buf_N(engine)
#     np.copyto(inputs[0].host, img.ravel())
#     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#     stream.synchronize()
#     trt_outs = [out.host for out in outputs]
#     print(np.array(trt_outs).reshape(128,10).shape)
    # print(trt_outs[0].shape)
    # print(trt_outs[1].shape)

# 
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
index = 6
def load_data(path, shape = (64,64)):
    data = list()
    label = list()
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()]
    )
    for i in path:
        image = Image.open(i)
        image = transform(image)
        data.append(image)
        label.append(index)
    return data, label

path =  sorted(glob.glob('/home/timtran/Desktop/TensorRT/20220331_ADLT_MergeLegacy_Train/%s/*'%(str('6'))))
x_data, x_label = load_data(path, shape = (64,64))
train_data = []
for i in range(len(x_data)):
   train_data.append([x_data[i], x_label[i]])
# print(train_data)
data = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0)

# # INFERENCE
with open('/home/timtran/Desktop/TensorRT/char_adlt-tf16.engine', "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
print(f'{bcolors.OKCYAN}' + 'Loaded Engine')

# data = glob.glob('/home/timtran/Desktop/TensorRT/20220331_ADLT_MergeLegacy_Train/0/*')
correct =0
for i, (inputs, labels) in enumerate(data, 0):
    
    ta = time.time()
    inputs = np.array(inputs)
    labels = np.array(labels)
    context.set_binding_shape(0, inputs.shape) 

    batch_size = inputs.shape[0]
    # inputs = imgs.astype(np.float32)
    inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine)

    # inputs_alloc_buf[0].host = np.ascontiguousarray(inputs)
    inputs_alloc_buf[0].host = inputs

    trt_outputs = do_inference_v2(engine, context, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf, stream_alloc_buf)
    trt_outputs = np.array(trt_outputs).reshape(128,37)

    output = np.argmax(trt_outputs.squeeze(), axis=1)[:batch_size]
    
    correct += np.sum(output == labels)

    # trt_output = trt_outputs[0].reshape(1, -1, num_classes)
    tb = time.time()
    print('time: %f secs' % (tb - ta))
    print('-----------------------------------')

print(correct/len(data.dataset))
# /home/terrychan/ADLT/TrainingData