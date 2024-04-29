import torch
import torch.nn as nn
from torch.autograd import Variable

import math

class SelfAttention(nn.Module):
	"""
	If we have a embedding of size embed_size we would be splitting it into heads
	Say we have embed size 256 and 8 heads, we would split it into 8*32 parts
	"""
	def __init__(self,embed_size,heads):
		super(SelfAttention,self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads

		assert (self.head_dim * heads == embed_size),"Embed size needs to be divisible"
		
		self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)
		self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
		self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)
		self.fc_out = nn.Linear(heads*self.head_dim,embed_size) # join outputs from  all heads

	def forward(self,values,keys,query,mask):
		N = query.shape[0] # no of examples we will send at one time
		value_len,key_len,query_len = values.shape[1],keys.shape[1],query.shape[1]

		# Split embedding into self.heads pieces
		values = values.reshape(N,value_len,self.heads,self.head_dim) # the reshape(self.heads,self.head_dim) splits the value tensor into multi heads (one part for each head)
		queries = query.reshape(N,query_len,self.heads,self.head_dim)
		keys = keys.reshape(N,key_len,self.heads,self.head_dim)

		values = self.values(values)
		queries = self.queries(queries)
		keys = self.keys(keys)

		# now we combine the outputs from the q,k,v's
		energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])

		# queries shape: (N,query_len,heads,heads_dim)
		# keys shape: (N,key_len,heads,heads_dim)
		# energy shape: (N,heads,query_len,key_len)

		"""
		To interpreat the shape of the query keys multiplied op we can consider the following:

		1. We have N inputs
		2. For each of N inputs we have 'heads' heads
		3. For each head we have a 'query_len' queries
		4. For each query we have a 'key_len' key
		
		If we consider the query (query_len) as the target sentence and the key (key_len) as the source sentence
		Then  the energy tells us how much attention we provide to each word of the input
		For self attention we take target sentence = source sentence
		""" 

		# sending in the mask
		# if an element is masked we omit that while computing the attention (should be a matrix)

		if mask is not None:
			energy = energy.masked_fill(mask==0,float("-1e20")) # replace 0s with -inf

		attention = torch.softmax(energy/(self.embed_size ** (1/2)),dim=3) # apply on the keys

		out = torch.einsum("nhqk,nvhd -> nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)

		# attention shape = (N,heads,query_len,key_len) # basicallly same shape as energy as we are only scaling it
		# values shape = (N,value_len,heads,head_dim)
		# out shape = (N,query_len,heads,head_dim)
		# flatten the last two dimensions of out (the einsum output)
		# p.s we know that the value len and key len are the same(enfored by how we send them from encoder),so multiply accross that dim

		"""

		We can think of the output as this way:

		1. We have n inputs
		2. For each of the N input we have a 'query len' queries
		3. For each query we have 'heads' heads
		4. For each head we have a value of dim 'head dim'

		This value spat out at last can be considered as the complete attention score computed via each head 
		and then they are concatenated via a reshape
		"""

		# lastly pass the output through a fully connected layer

		out = self.fc_out(out)

		return out

class TransformerBlock(nn.Module):
	def __init__(self,embed_size,heads,dropout,forward_expansion):
		super(TransformerBlock,self).__init__()

		self.attention = SelfAttention(embed_size,heads)
		self.norm1 = nn.LayerNorm(embed_size) # takes a norm for every single example other than the whole batch
		self.norm2 = nn.LayerNorm(embed_size)

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size,forward_expansion*embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion*embed_size,embed_size)
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self,value,key,query,mask):
		attention = self.attention(value,key,query,mask)

		x = self.dropout(self.norm1(attention+query))
		forward = self.feed_forward(x)
		out = self.dropout(self.norm2(forward+x))

		return out


class PositionalEncoder(nn.Module):
	def __init__(self,embed_size,max_length,dropout):
		super(PositionalEncoder,self).__init__()
		self.embed_size = embed_size
		self.max_length = max_length
		self.dropout = nn.Dropout(dropout)

		pe = torch.zeros(max_length,embed_size) # we have a vector of embed size
		for pos in range(max_length):
			for i in range(0,embed_size,2):
				# take a step of 2 from 0 to embed size
				pe[pos,i] = math.sin(pos/(10000**((2*i)/embed_size)))
				pe[pos,i+1] = math.cos(pos/(10000**((2*i)/embed_size)))
		
		pe = pe.unsqueeze(0)

		self.register_buffer("pe",pe)

	def forward(self,x):
		x = x * math.sqrt(self.embed_size)
		seq_len = x.size(1)
		pe = Variable(self.pe[:,:seq_len],requires_grad=False)
		if x.is_cuda:
			pe.cuda()
		
		x = x + pe

		return self.dropout(x)




class Encoder(nn.Module):
	def __init__(self,src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
		super(Encoder,self).__init__()
		self.embed_size = embed_size
		self.device = device
		self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
		self.position_mapper = nn.Embedding(max_length,embed_size)
		self.position_embedding = PositionalEncoder(embed_size,max_length,dropout)
		self.layers = nn.ModuleList(
			[
				TransformerBlock(
					embed_size,
					heads,
					dropout=dropout,
					forward_expansion=forward_expansion
				)
				for _ in range(num_layers)

			]
		)

		self.dropout = nn.Dropout(dropout)


	def forward(self,x,mask):
		N,seq_length = x.shape
		positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
		positions = self.position_mapper(positions)
		out = self.dropout(self.word_embedding(x)+self.position_embedding(positions))
		for layer in self.layers:
			out = layer(out,out,out,mask)

		return out

class DecoderBlock(nn.Module):
	def __init__(self,embed_size,heads,forward_expansion,dropout,device):
		super(DecoderBlock,self).__init__()
		self.attention = SelfAttention(embed_size,heads)
		self.norm = nn.LayerNorm(embed_size)
		self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion)

		self.dropout = nn.Dropout(dropout)

	def forward(self,x,value,key,src_mask,trg_mask):
		attention = self.attention(x,x,x,trg_mask)
		query = self.dropout(self.norm(attention+x))
		out = self.transformer_block(value,key,query,src_mask)
		return out



class Decoder(nn.Module):
	def __init__(self,trg_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,device,max_length):
		super(Decoder,self).__init__()
		self.device=device
		self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
		self.position_mapper = nn.Embedding(max_length,embed_size)
		self.position_embedding = PositionalEncoder(embed_size,max_length,dropout)
		self.layers = nn.ModuleList(
			[
				DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
				for _ in range(num_layers)
			]
		)

		self.fc_out = nn.Linear(embed_size,trg_vocab_size)
		self.dropout = nn.Dropout(dropout)


	def forward(self,x,enc_out,src_mask,trg_mask):
		N,seq_length = x.shape
		positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
		positions = self.position_mapper(positions)
		x = self.dropout((self.word_embedding(x)+self.position_embedding(positions)))
		for layer in self.layers:
			x = layer(x,enc_out,enc_out,src_mask,trg_mask)

		out = self.fc_out(x)

		return out


class Transformer(nn.Module):
	def __init__(self,src_vocab_size,trg_vocab_size,src_pad_index,trg_pad_index,embed_size=256,num_layers=6,forward_expansion=4,
					  heads=8,dropout=0,device = "cuda",max_length=100
				):

		super(Transformer,self).__init__()
		self.encoder = Encoder(src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length)

		self.decoder = Decoder(trg_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,device,max_length)

		self.src_pad_index = src_pad_index
		self.trg_pad_index = trg_pad_index 
		self.device = device

	def make_src_mask(self,src):
		src_mask = (src!= self.src_pad_index).unsqueeze(1).unsqueeze(2)

		#(N,1,1,src_len)

		return src_mask.to(device)

	def make_trg_mask(self,trg):
		N,trg_len = trg.shape
		trg_mask  = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)
		return trg_mask.to(self.device)


	def forward(self,src,trg):
		src_mask = self.make_src_mask(src)
		trg_mask = self.make_trg_mask(trg)

		enc_src = self.encoder(src,src_mask)
		out = self.decoder(trg,enc_src,src_mask,trg_mask)

		return out


# toy example to check if it runs
if __name__ == "__main__":
	device = "cuda"
	x = torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
	trg = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)

	src_pad_index = 0
	trg_pad_index = 0
	src_vocab_size = 10
	trg_vocab_size = 10

	model = Transformer(src_vocab_size,trg_vocab_size,src_pad_index,trg_pad_index).to(device)
	out = model(x,trg[:,:-1])
	print("The output shape from the transformer is: ",out.shape)




