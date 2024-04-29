from torch.autograd import Variable
import torch.onnx
import torchvision

from model import EncoderCNN, DecoderRNN, DecoderRNN_onnx
import pdb

dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()

encoder = EncoderCNN(256, 'resnet18').cuda()
#encoder.load_state_dict(torch.load('bin_models/resnet18/encoder-5-3000.ckpt'))
decoder = DecoderRNN_onnx(256, 512, 9957, 1).cuda()
#decoder.load_state_dict(torch.load('bin_models/resnet18/decoder-5-3000.ckpt'))

x = encoder(dummy_input)

torch.onnx.export(encoder, dummy_input, "encoder.onnx", verbose=True, export_params=True)
torch.onnx.export(decoder, x, "decoder.onnx", verbose=True, export_params=True)
