from torch import nn
from torch._jit_internal import weak_module, weak_script_method
from torch.nn import functional as F


@weak_module
class NormstabilizerLoss(nn.NLLLoss):
	"""
	The norm-stabilizer was introduced by David et.al. 
	Adds a term to the loss. 

	The forward takes a tensor `hidden_layers` which represents all the hidden states of 
	a single sequence. 

	`hidden_layers` is expected to be a tensor with dimensions: (sequence_length, batch_size, hidden_layer_size)

	Currently doesn not process Sequences. 
	"""


	__constants__ = ['ignore_index', 'weight', 'reduction','norm_stabilizer_param']

	def __init__(self, weight=None, size_average=None, ignore_index=-100,
				 reduce=None, reduction='mean',norm_stabilizer_param=None):
		super(Normstabilizer, self).__init__(weight, size_average, reduce, reduction)
		self.ignore_index = ignore_index
		self.norm_stabilizer_param = norm_stabilizer_param

	@weak_script_method
	def forward(self, input, target,hidden_layers):
		'''
		Expecting hidden_layers in the (sequence_length, batch_size, hidden_size)
		
		The paper defines the additional loss term as: 

			beta*(1/T) Σ(||ht||2 - ||ht-1||2)^2
		
		Where T is the sequence length, t is the time step and ht is the hidden state at timestep t

		After taking the norm the tensor condenses to (seq_len,batch_size), each value representing the l2 norm of a hidden state.
		
		We roll this vector and describe this term as the MSELoss where the rolled vector is the target and 
		the original vector is the input. 
		'''
		sequence_length = hidden_layers.shape[0]




		# h_y has shape (sequence_length, batch_size)
		# Each value being the l2 norm of a hidden state. 
		h_x = hidden_layers.norm(dim=2)
		h_y = torch.roll(h_y,-1,0)
		
		h_y = h_y[:-1].flatten()
		h_x = h_x[:-1].flatten()


		#default nll_loss. Expecting log softmax output. 
		_nllloss = F.nll_loss(input,target,weight=self.weight,ignore_index=self.ignore_index,reduction=self.reduction)
		

		_normstabilizer_loss = F.mse_loss(h_x,h_y,size_average=False,reduce=True,reduction='sum')


		#possible float correction
		_hyperparam = self.norm_stabilizer_param/sequence_length

		return _nllloss + _hyperparam*_normstabilizer_loss 

