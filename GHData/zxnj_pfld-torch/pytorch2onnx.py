import onnx
import os
import argparse
import torch
from torch.autograd import Variable
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--backbone', default="ghost")
parser.add_argument('--torch_model', default="")
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./output/v3.onnx")
args = parser.parse_args()

if args.backbone == "v2":
    from models.pfld import PFLDInference, AuxiliaryNet
elif args.backbone == "v3":
    from models.mobilev3_pfld import PFLDInference, AuxiliaryNet
elif args.backbone == "ghost":
    from models.ghost_pfld import PFLDInference, AuxiliaryNet
elif args.backbone == "lite":
    from models.lite import PFLDInference, AuxiliaryNet
else:
    raise ValueError("backbone is not implemented")

print("=====> load pytorch checkpoint...")
pfld_backbone = PFLDInference()
pfld_backbone.eval()
pfld_backbone.load_state_dict(torch.load(args.torch_model, map_location=torch.device('cpu'))['pfld_backbone'])
#print("PFLD bachbone:", pfld_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(
    pfld_backbone,
    dummy_input,
    args.onnx_model,
    verbose=True,
    opset_version=11,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"input": {0:"batch"}, "output":{0:"batch"}})


print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

#print("====> Simplifying...")
#model_opt = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
#onnx.save(model_opt, args.onnx_model_sim)
#print("onnx model simplify Ok!")
