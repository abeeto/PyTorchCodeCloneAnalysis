import torch


class FunctionSepconv(torch.autograd.Function):
	def __init__(self):
		super(FunctionSepconv, self).__init__()
	# end


	def forward(self, teach, module):#10x1
		self.save_for_backward(teach, module)
		module = module * 10
		return module
	# end
#####逆伝搬
	def backward(self, gradOutput):
		teach, module = self.saved_tensors
		gradOutput.backward()
		return gradOutput
	# end
# end
