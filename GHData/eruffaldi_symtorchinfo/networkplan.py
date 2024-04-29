#https://github.com/TylerYep/torchinfo
import sys
import torch as otorch
import torch.utils 

sys.modules["torch"] = sys.modules[__name__]
sys.modules["torch.nn"] = sys.modules[__name__]
utils = otorch.utils

from typing import Iterable, Any, SupportsFloat, Tuple, TypeVar, Union
import sympy as sp
from sympy import floor
from typing import NewType
import abc
import argparse

oint = int
ofloat=float
obool = bool
nn = sys.modules[__name__]
onnx = otorch.onnx
uint8 = otorch.uint8
int8 = otorch.int8
int16 = otorch.int16
int32 = otorch.int32
half =otorch.half
float =otorch.float
float32 =otorch.float32
float64 =otorch.float64
double =otorch.double
complex32 =otorch.complex32
complex64 =otorch.complex64
complex128 =otorch.complex128
int64 = otorch.int64
short = otorch.short
int = otorch.int
bool = otorch.bool
dtype = otorch.dtype
spExpressison1i = NewType('spExpressison1i', Any)
spExpressison2i = NewType('spExpressison2i', Any)

def check1i(x):
	return x

def checkExpand2i(x):
	m = sp.Matrix((x,))
	if m.shape == (1,1):
		return sp.Matrix((x,x))
	else:
		return m

class Tensor:
	def __init__(self,shape):
		self.shape = shape
	def flatten(self,d=1):
		# TBD
		return Tensor(prod(shape))
	def flatten_not1(self):
		return Tensor([shape[0],prod(shape[1:])])
	def __add__(self,other):
		return self

def flatten(x,d):
	return x.flatten(d)

#register_forward_pre_hook
#register_forward_hook

class Module(metaclass=abc.ABCMeta):
	def __init__(self):
		self.variables = {}
		self.name = ""
		self.children = []
		self.hook = None
		self.hookpre = None
	def parameters(self):
		return self.variables
	@abc.abstractmethod
	def forward(self,x):
		pass
	def modules(self):
		return []
	def __call__(self,*args,**kwargs):
		if self.hookpre is not None:
			self.hookpre(self,args=args,kwargs=kwargs)
		x = self.forward(*args,**kwargs)
		if self.hook is not None:
			self.hook(self,ret=x,args=args,kwargs=kwargs)
		return x





class ModuleList(Module):
	def __init__(self,*args):
		self.variables = {}
		self.name = ""
		self.children = args
	def modules(self):
		return self.children
	def forward(self,*args,**kwargs):
		return None


class ModuleDict(Module):
	def __init__(self,d):
		self.variables = {}
		self.name = ""
		self.children = d
	def modules(self):
		return self.children
	def forward(self,*args,**kwargs):
		return None
#BatchNorm2d
#MaxPool2d
#activation_func

class Sequential(Module):
	# TBD OrderedDict
	def __init__(self,*args):
		Module.__init__(self)
		self.children = args
	def forward(self,x):
		for b in self.children:
			x = b(x)
		return x
	def modules(self):
		return self.children

class SingleInputNet(Sequential):
	def __init__(self,*args):
		Sequential.__init__(self)


class Activation(Module):
	def __init__(self,inplace=True):
		Module.__init__(self)
	def forward(self,x):
		return Tensor(x.shape)

class BatchNorm2d(Module):
	def __init__(self,planes):
		Module.__init__(self)
	def forward(self,x):
		return Tensor(x.shape)

class Dropout2d(Module):
	def __init__(self):
		Module.__init__(self)
	def forward(self,x):
		return Tensor(x.shape)


class MaxPool2d(Module):
	def __init__(self,kernel_size=3, stride=2, padding=1):
		Module.__init__(self)
	def forward(self,x):
		return Tensor(x.shape)

class AdaptiveAvgPool2d(Module):
	def __init__(self,kernel_size=3, stride=2, padding=1):
		Module.__init__(self)
	def forward(self,x):
		return Tensor(x.shape)


class ReLU(Activation):
	def __init__(self,inplace=True):
		Activation.__init__(self,inplace=inplace)

class Linear(Module):
	def __init__(self,in_features, out_features, bias=True):
		Module.__init__(self)
		self.in_features = in_features
		self.out_features = out_features
		self.variables = dict(weight=(out_features,in_features))
		if bias:
			self.variables["bias"]=out_features

	def forward(self,x):	
		# TBD verify flattening of x right	
		return Tensor([y.shape[0],self.out_features])



class Conv1d(Module):
	#https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
	# (N,Cin,L) -> (N,out_channels,Lout)
	#
	#
	#groups controls the connections between inputs and outputs
	#	When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a depthwise convolution
	#
	#padding=valid is zero
	#padding=same (only with stride=1)
	#
	#
	# L_out = lower((L_in + 2*padding - dilator*(kernel_size-1) - 1))/stride + 1)
	#variables
	#	bias (out_channels)
	#	weight is (in_channels, in_channels/groups, kernel_size)	
	#
	# Autopadding is: (self.kernel_size[0] // 2 )
	def __init__(self, in_channels: spExpressison1i, out_channels: spExpressison1i, kernel_size: spExpressison1i, stride: spExpressison1i = 1,padding: Union[oint,spExpressison1i,str] = 0,dilation:spExpressison1i = 1,groups:spExpressison1i=1,bias=True,padding_mode="zeros",device=None,type=None):
		Module.__init__(self)
		self.variables = dict(bias=out_channels, weight=(out_channels,in_channels/groups,kernel_size))
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
		self.out_channels = out_channels
		self.in_channels = in_channels
		if padding == "same":
			if self.stride != 1:
				raise Error("Expected stride=1 for padding=same")
	def parameters(self):
		return self.variables

	def forward(self,data: Tensor):
		#assert data.shape size == 3
		N = data.shape[0]
		Cin = data.shape[1]
		# assert Cin == in_channels
		Lin = data.shape[2]	
		if self.padding == "valid":
			padding = 0	
		elif self.padding == "same":
			pass
		else:
			padding = self.padding
		Lout = floor(((Lin + 2*padding-self.dilation*(self.kernel_size-1)-1))/self.stride + 1)
		return Tensor([N,self.out_channels,Lout])


class Conv2d(Module):
	#https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
	# (N,Cin,L) -> (N,out_channels,Lout)
	#
	#
	#groups controls the connections between inputs and outputs
	#	When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a depthwise convolution
	#
	#padding=valid is zero
	#padding=same (only with stride=1)
	#
	#
	# L_out = lower((L_in + 2*padding - dilator*(kernel_size-1) - 1))/stride + 1)
	#variables
	#	bias (out_channels)
	#	weight is (in_channels, in_channels/groups, kernel_size)	
	#
	# Autopadding is: (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 
	def __init__(self, in_channels: spExpressison1i, out_channels: spExpressison1i, kernel_size: spExpressison2i, stride: spExpressison2i = 1,padding: Union[oint, str, spExpressison2i] = 0,dilation:spExpressison1i = 1,groups:spExpressison1i=1,bias=True,padding_mode="zeros",device=None,type=None):
		Module.__init__(self)
		stride = checkExpand2i(stride)
		kernel_size = checkExpand2i(kernel_size)
		dilation = checkExpand2i(dilation)
		groups = check1i(groups)
		if padding is not None or type(padding) is not str:
			padding = checkExpand2i(padding)
		self.variables = dict(bias=out_channels, weight=(out_channels,in_channels/groups,kernel_size[0],kernel_size[1]))
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
		self.out_channels = out_channels
		self.in_channels = in_channels
		if padding == "same":
			if self.stride != 1:
				raise Error("Expected stride=1 for padding=same")

	def forward(self,data: Tensor):
		#assert data.shape size == 4
		N = data.shape[0]
		Cin = data.shape[1]
		H = data.shape[2]	
		W = data.shape[3]	
		if self.padding == "valid":
			padding = 0	
		elif self.padding == "same":
			pass
		else:
			padding = self.padding
		Hout = floor(((H + 2*padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1))/self.stride[0] + 1)
		Wout = floor(((W + 2*padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1))/self.stride[1] + 1)
		return Tensor([N,self.out_channels,Hout,Wout])


class Conv3d(Module):
	#https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
	# (N,Cin,L) -> (N,out_channels,Lout)
	#
	#
	#groups controls the connections between inputs and outputs
	#	When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a depthwise convolution
	#
	#padding=valid is zero
	#padding=same (only with stride=1)
	#
	#
	# L_out = lower((L_in + 2*padding - dilator*(kernel_size-1) - 1))/stride + 1)
	#variables
	#	bias (out_channels)
	#	weight is (in_channels, in_channels/groups, kernel_size)	
	#
	# Autopadding is: (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 
	def __init__(self, in_channels: spExpressison1i, out_channels: spExpressison1i, kernel_size: spExpressison2i, stride: spExpressison2i = 1,padding: Union[oint, str, spExpressison2i] = 0,dilation:spExpressison1i = 1,groups:spExpressison1i=1,bias=True,padding_mode="zeros",device=None,type=None):
		Module.__init__(self)
		stride = checkExpand2i(stride)
		kernel_size = checkExpand2i(kernel_size)
		dilation = checkExpand2i(dilation)
		groups = check1i(groups)
		if padding is not None or type(padding) is not str:
			padding = checkExpand2i(padding)
		self.variables = dict(bias=out_channels, weight=(out_channels,in_channels/groups,kernel_size[0],kernel_size[1]))
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
		self.out_channels = out_channels
		self.in_channels = in_channels
		if padding == "same":
			if self.stride != 1:
				raise Error("Expected stride=1 for padding=same")

	def forward(self,data: Tensor):
		#assert data.shape size == 4
		N = data.shape[0]
		Cin = data.shape[1]
		H = data.shape[2]	
		W = data.shape[3]	
		if self.padding == "valid":
			padding = 0	
		elif self.padding == "same":
			pass
		else:
			padding = self.padding
		Hout = floor(((H + 2*padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1))/self.stride[0] + 1)
		Wout = floor(((W + 2*padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1))/self.stride[1] + 1)
		return Tensor([N,self.out_channels,Hout,Wout])

class ConvTranspose2d(Conv2d):
	def __init__(self, *args,**kwargs):
		Conv2d.__init__(*args,**kwargs)

class AccumSummary:
	def __init__(self):
		self.variables = []
		self.nodes = []
	def scan(self,model, input_size,depth,**kwargs):
		self.apply_hooks(model)
		x = Tensor(input_size)
		if isinstance(x, (list,tuple)):
			y = model(*x,**kwargs)
		elif isinstance(x,dict):
			y = model(**x,**kwargs)
		else:
			y = model(x,**kwargs)
		self.remove_hooks(model)
		return y
	def apply_hooks(self,model):
		print("hook",model.__class__.__name__)
		model.hookpre = lambda mod,args,kwargs: self.hookpre(mod,args,kwargs)
		model.hook =  lambda mod,ret,args,kwargs: self.hook(mod,ret,args,kwargs)
		for m in model.modules():
			self.apply_hooks(m)

	def remove_hooks(self,model):
		model.hookpre = None
		model.hook =  None
		for m in model.modules():
			self.remove_hooks(m)
	def hookpre(self,module,args,kwargs):
		pass
	def hook(self,module,ret,args,kwargs):
		id = module.name
		if id == "":
			id = "%s#%s" % (module.__class__.__name__,"")
		print("hook",id)
		for k,v in module.parameters().items():
			self.variables.append(dict(name="%s_%s" % (id,k),params=v))
		#self.nodes.append(dict(op=model,input=data.shape,output=z.shape,variables={}.update(model.variables)))

"""
================================================================================================================
Layer (type:depth-idx)          Input Shape          Output Shape         Param #            Mult-Adds
================================================================================================================
SingleInputNet                  --                   --                   --                  --
├─Conv2d: 1-1                   [7, 1, 28, 28]       [7, 10, 24, 24]      260                1,048,320
├─Conv2d: 1-2                   [7, 10, 12, 12]      [7, 20, 8, 8]        5,020              2,248,960
├─Dropout2d: 1-3                [7, 20, 8, 8]        [7, 20, 8, 8]        --                 --
├─Linear: 1-4                   [7, 320]             [7, 50]              16,050             112,350
├─Linear: 1-5                   [7, 50]              [7, 10]              510                3,570
================================================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
Total mult-adds (M): 3.41
================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 0.40
Params size (MB): 0.09
Estimated Total Size (MB): 0.51
================================================================================================================
"""
def summary(model, input_size,depth=0,**kwargs):
	ac = AccumSummary()
	output = ac.scan(model, input_size,depth,**kwargs)
	if output is not None:
		print("outputsize",output.shape)
	print("Variables")
	for v in ac.variables:
		print(v)

class OpsTorchVision:
	def __init__(self):
		pass
	def _cuda_version(self):
		return 0
class Ops:
	def __init__(self):
		self.torchvision = OpsTorchVision()
	def load_library(self,name):
		print("loading",name)

class TorchVersion():
	def __init__(self):
		self.cuda = None

def init(*args,**kwargs):
	print("Init")

class Functional:
	def __init__(self):
		self.interpolate = None

class Jit:
	def __init__(self):
		self.unused = lambda x: x
		self._overload_method = lambda x:x
		self._script_if_tracing = lambda x:x
class Function:
	pass
class Autograd:
	def __init__(self):
		self.Function = Function
class CGraph:
	def __init__(self):
		self.op = None
class CBlock:
	def __init__(self):
		self.op = None
class CNode:
	def __init__(self):
		self.op = None
class CUnk:
	def __init__(self):
		self.Block = CBlock()
		self.Graph = CGraph()
		self.Node = CNode()

class Device:
	def __init__(self,what):
		return
	def __call__(self,what):
		return Device(what)

def no_grad():
	return lambda x: x	
device = Device("cpu")


_C = CUnk()
autograd = Autograd()
jit = Jit()
ops = Ops()
version = TorchVersion()
functional = Functional()

def main():
	#nn = sys.modules[__name__]
	parser = argparse.ArgumentParser(description='Evaluate models')
	parser.add_argument("mode")
	#parser.add_argument('output')
	#parser.add_argument('-x',action="store_true")
	args = parser.parse_args()
	mode = args.mode
	N, Lin,a,b,c = sp.symbols('N, Lin, a, b, c')
	if mode == "conv1":
		model = nn.Conv1d(a, 33, 3, stride=2)
		summary(model,(N, a, Lin))
	elif mode == "conv2":
		print("complex")
		m = nn.Conv2d(16, b, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
		summary(m,(N, a, 50,16))
		if False:
			print("\nsimpler")
			m = nn.Conv2d(a, 33, 3, stride=2)
			summary(m,(N, a, 50,16))
			m = nn.Conv2d(a, 33, (3, 5), stride=(2, 1), padding=(4, 2))
			summary(m,(N, a, 50,16))
	elif mode == "seq":
		model = nn.Sequential(
          nn.Conv2d(a,b,5),
          nn.ReLU(),
          nn.Conv2d(b,c,5),
          nn.ReLU()
        )
		summary(model,(N, a, 50,16))
	elif mode == "convnet":
		model = SingleInputNet([Conv2d(1,10),Conv2d(10,20),Droput2d(),Linear(320,50),Linear(50,10)])
		batch_size = 16
		summary(model, input_size=(batch_size, 1, 28, 28))
	elif mode == "resnet":
		import torchvision
		model = torchvision.models.resnet152()
		summary(model, (1, 3, 224, 224), depth=3)
	else:
		print("unknown model")


if __name__ == '__main__':
	main()




