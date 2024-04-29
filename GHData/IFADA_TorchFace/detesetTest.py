import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import nets
from tool import utils

class Deteset:
	def __init__(self, pnet_param=r'D:\Pyproject\TorchFace\param\pnet.pkl',
	             rnet_param=r'D:\Pyproject\TorchFace\raram\rnet.pkl',
	             onet_param=r'D:\Pyproject\TorchFace\oaram\onet.pkl', isCuda=True):
		self.isCuda = isCuda
		self.pnet = nets.Pnet()
		self.rnet = nets.Rnet()
		self.onet = nets.Onet()

		if self.isCuda:
			self.pnet.cuda()
			self.rnet.cuda()
			self.onet.cuda()

		self.pnet.load_state_dict(torch.load(pnet_param))
		self.rnet.load_state_dict(torch.load(rnet_param))
		self.onet.load_state_dict(torch.load(onet_param))

		self.pnet.eval()
		self.rnet.eval()
		self.onet.eval()

		self._image_transform = transforms.Compose([transforms.ToTensor()])


	def deteset(self, image):

		pnet_boxes = self._pnet_deteset(image)
		if pnet_boxes.shape[0] ==0:
			return np.array([])

	def _pnet_deteset(self,image):
		boxes=[]
		img =image
		h,w = img.size
		min_side= min(h,w)
		scale =1
		while min_side>12:
			img_data =self._image_transform(img)
			if self.isCuda:
				img_data = img_data.cuda()
			img_data.unsqueeze(0)
			_cls,_offset = self.pnet(img)
			cls = _cls[0][0].cpu().data
			offset = _offset[0].cpu().data
			idxs= torch.nonzero(torch.gt(cls,0.6))

			for idx in idxs:
				boxes.append(self._box(idx,offset,cls[idx[0],idx[1]],scale))
			scale *=0.7
			_w = int(w*scale)
			_h = int(h * scale)
			min_side = min(_w,_h)
		return utils.nms(np.array(boxes),0.5)




	def _box(self,start_index,offset,cls,scale,stride =2,side_len =12):
		#建议框
		start_index = np.array(start_index,dtype=np.float32)
		_x1 = (start_index[1]*stride)/scale
		_y1 = (start_index[0]*stride)/scale
		_x2 = (start_index[1]*stride+side_len)/scale
		_y2 = (start_index[0]*stride+side_len)/scale
		#实际框
		ow = _x2 -_x1
		oh = _y2 -_y1
		_offset = offset[:,start_index[0],start_index[1]]
		x1 = _x1 + ow * _offset[0]
		y1 = _y1 + ow * _offset[1]
		x2 = _x2 + ow * _offset[2]
		y2 = _y2 + ow * _offset[3]

		return [x1, y1, x2, y2, cls]








