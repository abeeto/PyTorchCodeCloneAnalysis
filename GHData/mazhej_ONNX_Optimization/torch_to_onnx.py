import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from vision import *
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

#get all the pre trained model from torch
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))


#create dummy input _ for inception the input is 299,299 , yolo(416,416)
dummy_input = torch.randn(1,3,224,224)

#select our desired model
models_list=['resnet152','resnet50','squeezenet1_1','densenet169','mobilenet_v2','inception_v3']
for m in models_list:
    model =models.__dict__[m](pretrained=True, )
    model.cpu()
    model.eval()
    torch.onnx.export(model, dummy_input, f"/1TBstorage/OnnxModels_New/{m}.onnx", verbose=False,keep_initializers_as_inputs=True,opset_version=10)


#for comparing models vs quantization models
# model = models.resnet50(pretrained=True,)
# model.cpu()
# model.eval()
# torch.onnx.export(model,dummy_input,'/1TBstorage/torch_model_quantized/resnet50.onnx',verbose=False,keep_initializers_as_inputs=True,opset_version=10)

# # import torchvision.models.quantization as model
# model_q = model.resnet50(pretrained= True, quantize=False)
# torch.onnx.export(model_q,dummy_input,'/1TBstorage/torch_model_quantized/resnet50quantized.onnx',verbose=False,keep_initializers_as_inputs=True,opset_version=10)


##YOLO##
#from yolo_models import *

#model = Darknet('/home/maziar/WA/torch_to_onnx/yolo_cfg/yolov3-spp.cfg')
# # Load weights
# # attempt_download('yolo_weights/yolov3-tiny.weights.pt')
# # if 'yolo_weights/ultralytics68.pt'.endswith('.pt'):  # pytorch format
# #     model.load_state_dict(torch.load('yolo_weights/ultralytics68.pt', map_location='cpu')['model'])
#  # darknet format
# load_darknet_weights(model, 'yolo_weights/yolov3-spp.weights')

# model.cpu()
# model.eval()
#dummy_input = torch.randn(1,3,416,416)  
#out = model(dummy_input) 
#a=1
# torch.onnx.export(model,dummy_input,"/1TBstorage/OnnxModels_New/yolov3-spp.onnx", verbose=False,keep_initializers_as_inputs=True, opset_version=10)


#MOBILENET_SSD
# model = create_mobilenetv1_ssd(21, is_test=False)
# model.load('/home/maziar/WA/torch_to_onnx/mobilenet-v1-ssd-mp-0_675.pth')
# model.cpu()
# model.eval()

# dummy_input = torch.randn(1,3,300,300)
# torch.onnx.export(model,dummy_input,"/1TBstorage/OnnxModels_New/mobilenet_v1_ssd.onnx", verbose=False,keep_initializers_as_inputs=True, opset_version=10)


#DG_Models
# models_list=['resnet50_pruned_70_state_dict.pth','resnet50_pruned_83_state_dict.pth','resnet50_pruned_85_state_dict.pth']
# for m in models_list:
#     model= models.resnet50(pretrained=True)
#     model.load_state_dict(torch.load(f'/1TBstorage/DG_Models/{m}'))
#     model.cpu()
#     model.eval()
#     torch.onnx.export(model,dummy_input,f"/1TBstorage/OnnxModels_New/{m}.onnx", verbose=False,keep_initializers_as_inputs=True, opset_version=10)


##SSD
# precision = 'fp32'
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
# model.cpu()
# model.eval()
# dummy_input = torch.randn(1,3,300,300)
# torch.onnx.export(model,dummy_input,f"/1TBstorage/OnnxModels_New/SSD.onnx", verbose=False,keep_initializers_as_inputs=True, opset_version=10)




####segmentatin
model = models.segmentation.__dict__['deeplabv3_resnet101'](num_classes=21,
                                                                 aux_loss=True,
                                                                 pretrained=True)

model.cpu()
model.eval()
torch.onnx.export(model, dummy_input, f"/1TBstorage/OnnxModels_New/deeplabv3_resnet101.onnx", verbose=False,keep_initializers_as_inputs=True,opset_version=10)

