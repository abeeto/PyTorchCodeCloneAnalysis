import torch
import torchvision

from models.refinedet import build_refinedet
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_classes = 21                     # +1 for background
model = build_refinedet('test', 320, num_classes)            # initialize SSD
#net.load_state_dict(torch.load('/home/bmw/anaconda3/envs/refinedet37/RefineDet.PyTorch/weights/RefineDet320_VOC_5000.pth'))
model.load_state_dict(torch.load('/home/bmw/anaconda3/envs/refinedet37/RefineDet.PyTorch/weights/RefineDet320_VOC_5000.pth'))
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "num_classes", "size", "bkg_label", "top_k", "conf_thresh", "nms_thresh", "objectness_thre", "keep_top_k","arm_loc_data" , "arm_conf_data", "odm_loc_data", "odm_conf_data", "prior_data" ]
output_names = ["output"]

'''
num_classes = 21
size = 320
bkg_label = 0
top_k = 1000
conf_thresh = 0.01
nms_thresh = 0.45
objectness_thre = 0.01
keep_top_k = 500

arm_loc_data =  torch.randn(1, 6375, 4)
arm_conf_data = torch.randn(1, 6375, 2)
odm_loc_data = torch.randn(1, 6375, 4)
odm_conf_data = torch.randn(1, 6375, 21)
prior_data = torch.randn(6375, 4)


dummy_input = [num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh, objectness_thre, keep_top_k, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data]
'''
#dummy input defined with batch size = 32 and  image dimensions 3, 320, 320
dummy_input = torch.randn(32, 3, 320, 320)

torch.onnx.export(model, dummy_input, "/home/bmw/anaconda3/envs/refinedet37/RefineDet.PyTorch/refinedet.onnx", verbose=True, input_names=input_names, output_names=output_names)