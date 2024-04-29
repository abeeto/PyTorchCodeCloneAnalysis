import torch
from torch.autograd import Variable

batch_size = 100
row_lenth = 10
col_length = 10

if __name__ == '__main__':
	a = Variable(torch.randn((batch_size, row_lenth, col_length)))
	b = Variable(torch.randn((batch_size, row_lenth, col_length)))
	c = Variable(torch.randn((batch_size, row_lenth, col_length)))
	sum = Variable(torch.zeros((batch_size, row_lenth, col_length)))

	#for i in range(batch_size):
	#	sum[i, :, :] = a[i, :, :] + b[i, :, :]

	# 1. addition
	#d = torch.add(a, 1, b)
	#e = torch.add(a, b)

	# 2. tensor multiplication
	#d = torch.zeros((batch_size, row_lenth, col_length))
	#for i in range(batch_size):
	#	d[i, :, :] = torch.mm(a[i, :, :], b[i, :, :])
	#e = torch.matmul(a, b)
	#if torch.eq(d, e).sum() == 10000:
	#	print('true')

	# 3. element-wise multiplication
	#d = torch.FloatTensor([[1, 2], [3, 4]])
	#e = torch.FloatTensor([[1, 2], [3, 4]])
	#print(torch.addcmul(torch.zeros((2, 2)), value=1, tensor1=d, tensor2=e))
	#print(d * e)

	# 4. outer-product (vector, vector)
	#d = torch.LongTensor([1, 2, 3, 4, 5]).unsqueeze(0)
	#e = torch.mm(torch.t(d), d)
	#print(e)

	# 5. outer-product (matrix, matrix)
	#t = (batch_size, row_lenth, row_lenth)
	#d = torch.randn((batch_size, row_lenth)) 
	#e = torch.randn((batch_size, row_lenth))
	#f = d.unsqueeze(-1).expand(*t) * e.unsqueeze(-2).expand(*t) # (100, 10, 10)
	#print(f.size())
	#g = torch.randn((batch_size, row_lenth, row_lenth))
	#for i in range(batch_size):
	#	g[i, :, :] = torch.mm(d[i, :].unsqueeze(1), e[i, :].unsqueeze(0))
	#print(torch.eq(f, g).sum())























	

	
	

