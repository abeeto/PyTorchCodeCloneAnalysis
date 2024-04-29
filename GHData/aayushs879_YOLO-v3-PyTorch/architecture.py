from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np 
from util import *


def read_cfg(file):
	file = open(file, 'r')
	lines = file.read().split('\n')
	lines = [line for line in lines if len(line)>0]
	lines = [line for line in lines if line[0] != '#']
	lines = [line.rstrip().lstrip() for line in lines]


	block = {}
	blocks = []
	for line in lines:
		if line[0] == '[':
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block['type'] = line[1: -1].rstrip()

		else:
			key, value = line.split('=')
			block[key.rstrip()] = value.lstrip()

	blocks.append(block)

	return blocks



def create_modules(blocks):
	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3
	output_filters = []


	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()


		if (x['type'] == 'convolutional'):

			activation = x['activation']

			try:
				batch_normalize = int(x['batch_normalize'])
				bias = False

			except:
				batch_normalize = 0
				bias = True

			filters = int(x['filters'])
			padding = int(x['pad'])
			kernel_size = int(x['size'])
			stride = int(x['stride'])

			if padding:
				pad = (kernel_size - 1)//2
			else:
				pad = 0


			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
			module.add_module('conv_{0}'.format(index), conv)


			if batch_normalize:
				norm = nn.BatchNorm2d(filters)
				module.add_module('batch_norm_{0}'.format(index), norm)


			if activation == 'leaky':
				act = nn.LeakyReLU(0.1, inplace = True)
				module.add_module('leaky_{0}'.format(index), act)




		elif (x['type'] == 'upsample'):
			stride = int(x['stride'])
			upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
			module.add_module('upsample_{0}'.format(index), upsample)



		elif (x['type'] == 'route'):
			x['layers'] = x['layers'].split(',')

			start = int(x['layers'][0])


			try:
				end = int(x['layers'][1])
			except:
				end = 0 

			if start >0 :
				start = start - index

			if end > 0:
				end = end - index

			route = EmptyLayer()
			module.add_module('route_{0}'.format(index), route)


			if len(x['layers']) > 1:
				filters = output_filters[index + start] + output_filters[index + end]

			else:
				filters = output_filters[index + start]


		elif (x['type'] == 'shortcut'):
			shortcut = EmptyLayer()
			module.add_module('shortcut_{0}'.format(index), shortcut)


		elif (x['type'] == 'yolo'):
			mask = x['mask'].split(',')
			mask = [int(a) for a in mask]

			anchors = x['anchors'].split(',')
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]
			num_classes = int(net_info['classes'])
			img_dim = int(net_info['height'])
			detection = DetectionLayer(anchors, num_classes, img_dim)
			module.add_module('detection_{0}'.format(index), detection)

		module_list.append(module)
		prev_filters = filters
		output_filters.append(filters)


	return(net_info, module_list)




class EmptyLayer(nn.Module):
	def __init__(self):
		super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors, num_classes, img_dim):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.bbox_attrs = 5 + num_classes
		self.image_dim = img_dim
		self.ignore_thres = 0.5
		self.lambda_coord = 1

		self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
		self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
		self.ce_loss = nn.CrossEntropyLoss()  # Class loss

	def forward(self, x, targets = None):
		nA = self.num_anchors
		nB = x.size(0)
		nG = x.size(2)
		stride = self.image_dim / nG

		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
		ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

		prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
		x = torch.sigmoid(prediction[..., 0])  # Center x
		y = torch.sigmoid(prediction[..., 1])  # Center y
		w = prediction[..., 2]  # Width
		h = prediction[..., 3]  # Height
		pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
		pred_cls = torch.sigmoid(prediction[..., 5:])

		grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
		grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
		scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
		anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
		anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))


		pred_boxes = FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x.data + grid_x
		pred_boxes[..., 1] = y.data + grid_y
		pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
		pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

		if targets is not None:

			if x.is_cuda:
				self.mse_loss = self.mse_loss.cuda()
				self.bce_loss = self.bce_loss.cuda()
				self.ce_loss = self.ce_loss.cuda()

			nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
				pred_boxes=pred_boxes.cpu().data,
				pred_conf=pred_conf.cpu().data,
				pred_cls=pred_cls.cpu().data,
				target=targets.cpu().data,
				anchors=scaled_anchors.cpu().data,
				num_anchors=nA,
				num_classes=self.num_classes,
				grid_size=nG,
				ignore_thres=self.ignore_thres,
				img_dim=self.image_dim,
			)

			nProposals = int((pred_conf > 0.5).sum().item())
			recall = float(nCorrect / nGT) if nGT else 1
			precision = float(nCorrect / nProposals)

			# Handle masks
			mask = Variable(mask.type(ByteTensor))
			conf_mask = Variable(conf_mask.type(ByteTensor))

			# Handle target variables
			tx = Variable(tx.type(FloatTensor), requires_grad=False)
			ty = Variable(ty.type(FloatTensor), requires_grad=False)
			tw = Variable(tw.type(FloatTensor), requires_grad=False)
			th = Variable(th.type(FloatTensor), requires_grad=False)
			tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
			tcls = Variable(tcls.type(LongTensor), requires_grad=False)

			# Get conf mask where gt and where there is no gt
			conf_mask_true = mask
			conf_mask_false = conf_mask - mask

			# Mask outputs to ignore non-existing objects
			loss_x = self.mse_loss(x[mask], tx[mask])
			loss_y = self.mse_loss(y[mask], ty[mask])
			loss_w = self.mse_loss(w[mask], tw[mask])
			loss_h = self.mse_loss(h[mask], th[mask])
			loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
			loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
			loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

			return (
				loss,
				loss_x.item(),
				loss_y.item(),
				loss_w.item(),
				loss_h.item(),
				loss_conf.item(),
				loss_cls.item(),
				recall,
				precision,
			)

		else:
			output = torch.cat((pred_boxes.view(nB, -1, 4) * stride, pred_conf.view(nB, -1, 1),pred_cls.view(nB, -1, self.num_classes),),-1)
			return output





class Yolov3(nn.Module):
	def __init__(self, cfgfile, img_size):
		super(Yolov3, self).__init__()
		self.blocks = read_cfg(cfgfile)
		self.net_info, self.module_list = create_modules(self.blocks)
		self.img_size = img_size
		self.loss_names = ['x', 'y', 'w', 'h', 'objectness', 'class_prob', 'recall', 'precision']

	def forward(self, x, targets = None):
		detections = []
		modules = self.blocks[1:]
		layer_outputs = {}   #We cache the outputs for the route layer
		self.losses = defaultdict(float)
		output = []

		write = 0
		for i in range(len(modules)):        
		    
			module_type = (modules[i]["type"])
			if module_type == "convolutional" or module_type == "upsample":

				x = self.module_list[i](x)
				

    
			elif module_type == "route":
				layers = modules[i]["layers"]
				layers = [int(a) for a in layers]

				if (layers[0]) > 0:
					layers[0] = layers[0] - i

				if len(layers) == 1:
					x = layer_outputs[i + (layers[0])]

				else:
					if (layers[1]) > 0:
						layers[1] = layers[1] - i
					    
					map1 = layer_outputs[i + layers[0]]
					map2 = layer_outputs[i + layers[1]]


					x = torch.cat([map1, map2], 1)
				
            
			elif  module_type == "shortcut":
				from_ = int(modules[i]["from"])
				x = layer_outputs[i-1] + layer_outputs[i+from_]
				

			elif module_type == 'yolo':        
				if targets is not None:
					x, *losses = self.module_list[i][0](x, targets)
					for name, loss in zip(self.loss_names, losses):
						self.losses[name] += loss


				else:
					x = module(x)
				output.append(x)
			layer_outputs[i] = x

		self.losses['recall'] /= 3
		self.losses['precision'] /= 3
		return sum(output) if not (targets == None) else torch.cat(output, 1)






    

cuda = torch.cuda.is_available()

os.makedirs('output', exist_ok = True)

model = Yolov3('yolov3-voc.cfg', 416)

model.apply(init_weights)

if cuda:
	model = model.cuda()

model.train
