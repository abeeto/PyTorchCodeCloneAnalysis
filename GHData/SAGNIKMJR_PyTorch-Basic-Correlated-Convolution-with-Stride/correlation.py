import torch
import torch.nn as nn
import torch.optim as optim

class correlation_layer(nn.Module):

    def __init__(self, height, width, stride=1):
        super(correlation_layer, self).__init__()

        self.maskStridedCorrelation = torch.zeros(height, width)
        for i in range(height):
        	for j in range(width):
        		if i%stride==0 and j%stride==0:
        			self.maskStridedCorrelation[i,j]=1.


    def forward(self, x1, x2):
    	# exact same dimensionality for computing correlation
		assert x1.size(0) == x2.size(0)
		assert x1.size(1) == x2.size(1)
		assert x1.size(2) == x2.size(2)
		assert x1.size(3) == x2.size(3)

		# expanding mask for strided correlation to match dimensionality
		maskStridedCorrelation = self.maskStridedCorrelation.expand(x1.size(0), x1.size(1), 
										self.maskStridedCorrelation.size(0), self.maskStridedCorrelation.size(1))

		return (maskStridedCorrelation*x1)*x2

# network with a correlation layer and a linear layer
class network(nn.Module):

    def __init__(self, height, width, no_channels, num_classes, stride=1):
    	super(network, self).__init__()
    	self.corl1 = correlation_layer(height, width, stride) 
    	self.lin1 = nn.Linear(height*width*no_channels, num_classes)
    def forward(self, x1, x2):
    	correlation = self.corl1(x1, x2)
    	return self.lin1(correlation.view(x1.size(0), -1))


def main():

	batch_size = 32
	input_height = 100
	input_width = 75
	input_channels =224
	number_classes = 10
	stride = 2
	epochs = 3

	model = network(input_height, input_width, input_channels, number_classes, stride=stride)

	optimizer = optim.SGD(model.parameters(), lr = 0.01)

	input1 = torch.randn(batch_size,input_channels,input_height,input_width)
	input2 = torch.randn(batch_size,input_channels,input_height,input_width)
	for epoch in xrange(epochs):
	 	out = model(input1, input2)
	 	loss = torch.sum(out)
	 	loss.backward()
	 	optimizer.step()

	 	# replacing with new inputs
		input1 = torch.randn(batch_size,input_channels,input_height,input_width)
		input2 = torch.randn(batch_size,input_channels,input_height,input_width)

		print('epoch:{}, loss:{}'.format(epoch, loss))

if __name__=='__main__':
	main()