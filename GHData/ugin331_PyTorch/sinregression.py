# train nn for regression task
# use numpy to make sin wave --> logistic reg to predict??
# build training process to something more complex than linear(1, 1)
# do arithmetic and print out gradient
# pytorch tensor source
# torch.tensor
import inline
import matplotlib
import torch as torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torch.utils.data as data

import matplotlib.pyplot as plt

import numpy as np
import imageio

x_data = Variable(torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1))
y_data = Variable(torch.sin(x_data)+0.2*torch.rand(x_data.size()))

plt.figure(figsize=(10, 4))
plt.scatter(x_data.data.numpy(), y_data.data.numpy(), color="blue")
plt.title('Regression Analysis')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.savefig('curve_2.png')
plt.show()

class Net(torch.nn.Module):

   def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
       super(Net, self).__init__()
       self.hidden = torch.nn.Linear(n_feature, n_hidden)
       self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
       self.predict = torch.nn.Linear(n_hidden2, n_output)

   def forward(self, x):
       x = functional.leaky_relu(self.hidden(x))
       x = functional.leaky_relu(self.hidden2(x))
       x = self.predict(x)
       return x

   # our model


our_model = Net(n_feature=1, n_hidden=200, n_hidden2=100, n_output=1)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(our_model.parameters(), lr=0.05)

my_images = []
fig, ax = plt.subplots(figsize=(16, 10))

for epoch in range(1000):
   # Forward pass: Compute predicted y by passing
   # x to the model
   pred_y = our_model(x_data)

   # Compute and print loss
   loss = criterion(pred_y, y_data)

   # Zero gradients, perform a backward pass,
   # and update the weights.
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   print('epoch {}, loss {}'.format(epoch, loss.item()))

   if epoch % 10 == 0:
       # plot and show learning process
       plt.cla()
       ax.set_title('Regression Analysis - model 1', fontsize=35)
       ax.set_xlabel('Independent variable', fontsize=24)
       ax.set_ylabel('Dependent variable', fontsize=24)
       ax.set_xlim(-11.0, 13.0)
       ax.set_ylim(-1.1, 1.2)
       ax.scatter(x_data.data.numpy(), y_data.data.numpy(), color="blue")
       ax.plot(x_data.data.numpy(), pred_y.data.numpy(), 'g-', lw=3)
       ax.text(8.8, -0.8, 'Step = %d' % epoch, fontdict={'size': 24, 'color': 'red'})
       ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
               fontdict={'size': 24, 'color': 'red'})

       # Used to return the plot as an image array
       # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
       fig.canvas.draw()  # draw the canvas, cache the renderer
       image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
       image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

       my_images.append(image)

new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())

# save images as a gif
imageio.mimsave('./curve_2_model_1.gif', my_images, fps=10)
