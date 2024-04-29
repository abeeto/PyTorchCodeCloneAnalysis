import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

class EncoderRNN(nn.Module):
	'''
	构建编码器
	'''
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)    # 第一层embedding
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True, 
			num_layers = self.n_layers, bidirectional = True)     # 第二层GRU,可以由num_layers定义很多层

	def forward(self, input, hidden):                   # 前馈过程
		embedded = self.embedding(input)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

class AttnDecoderRNN(nn.Module):
	'''
	基于注意力的解码器
	'''
	def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length = MAX_LENGTH):
		······
		self.attn = nn.Linear(self.hidden_size * (2 * n_layers + 1), self.max_length)    # 注意力网络
		self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)    # 注意力机制作用完后映射到后面的层
		self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first = True, 
			num_layers = self.n_layers, bidirectional = True)     # 双向gru
		······
	def forward(self, input, hidden, encoder_outputs):
		······
		attn_weights = attn_weights.unsqueeze(1)         # 从注意力层取出权重大小，赋给attn_weights
		attn_applied = torch.bmm(attn_weights, encoder_outputs)   # 权重和输出相乘得到注意力计算结果
		output = self.attn_combine(output).unsqueeze(1)  # 注意力作用后的结果拼接成一个大的输入向量
		output = F.relu(output)                    # 将大的输入向量映射为gru的隐含层
		output, hidden = self.gru(output, hidden)  # 解码器gru运算
		# 取出gru运算最后一步的结果未给最后一层全联接层
		output = self.out(output[:, -1, :])
		output = F.log_softmax(output, dim = 1)
		return output, hidden, attn_weights
