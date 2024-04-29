import torch
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor

'''
     + Matrix Initializers
        Obv, the .type() is optional.

        - Initializers : All of these return torch.FloatTensor objects by default.

            torch.Tensor(r, c)              : Uninitialized
            torch.ones(r, c)
            torch.zeros(r, c)
            torch.rand(r, c)
            torch.randn(r, c)               : Random, following a normal distribution. 
            torch.eye(s) 

        - Types
            
            torch.FloatTensor
            torch.IntTensor
            torch.CharTensor
            torch.LongTensor
            torch.DoubleTensor

        - Methods
            
            tensor.size()                   : returns a tuple w/ matrix size. - can typecast to a list.
            tensor.numpy()                  : returns a numpy array of the tensor.
            torch.from_numpy(np_array)      : returns a tensor from numpy array.

'''

x = torch.Tensor(2, 4)
x = torch.rand(2, 3).type(dtype)
x = torch.randn(3, 4).type(torch.FloatTensor)
x = torch.ones(2, 2).type(torch.IntTensor)
x = torch.zeros(3, 4)
x = torch.eye(4)

'''
    + Mathematical Operations : All of these return a torch.FloatTensor object by default.

        torch.add(tensor, tensor)
        torch.t(tensor)                         : Transpose
        torch.abs(tensor)
        torch.clamp(input, min, max)            : min - (if input[] < min) | input[] | max : (if input[] > max)
        torch.div(tensor, value)                : division
        torch.exp(tensor)
        torch.log(tensor)
        torch.loglp(tensor)                     : log(1 + tensor)
        torch.mean(tensor)
        torch.mul(tensor, value/otherTensor)    : elementwise multiplication
        torch.matmul(tensor, tensor)
        torch.mm(tensor, tensor)                : matrix multiplication w/o broadcast
        torch.pow(tensor, value)
        torch.reciprocal(tensor)
        torch.remainder(tensor, devider)
        torch.round(tensor)
        torch.sigmoid(tensor)
        torch.sin()                             : and other trig. functions
        torch.sqrt()
        torch.sum()

    + In-Place operations

        tensor1.add_(tensor2)                   : tensor1 += tensor2
        tensor.t_()                             : tensor = torch.t(tensor)
        tensor.zero_()                          : tensor = torch.zeros()
'''

y = torch.ones(4, 2)
z = torch.mm(x, y)
z = torch.exp(z)
z.t_()


''' 
    + Autograd : When using autograd, the forward pass of your network will define a computational graph; 
        nodes in the graph will be Tensors, and edges will be functions that produce output 
        Tensors from input Tensors. Hence, even your placeholders have to be Variable(requires_grad=False).
        
        - Variable(tensor, requires_grad?) : For placeholders, requires_grad=False

        The computational graph in PyTorch is dynamic, ie. at each iteration a new graph is created.
        To use Autograd, we wrap around our Tensors in Variable. Each Variable has 2 properties, 
            .data : is the Tensor.
            .grad : is the gradient of it w/ respect to some scalar. (Both are of Tensor type)

        All other operations which you can perform on Tensors can be performed on Variables.

    + Essentials : For each iteration in the loop
        1. Calculate the forward pass
        2. Define the loss
        3. Calculate the gradients for all trainable Variables using :
            loss.backward()
        4. Update the weight's data by the gradients.

    + TODO : Defining new Autograd functions

'''

X = Variable(torch.randn(100, 50), requires_grad=False)
Y = Variable(torch.randn(100, 1), requires_grad=False)

W1 = Variable(torch.randn(50, 20), requires_grad=True)
W2 = Variable(torch.randn(20, 1), requires_grad=True)

learning_rate, no_steps = 1e-3, 100

for i in range(no_steps):
    a1 = torch.clamp(torch.mm(X, W1), min=0) # ReLU(X * W1)
    y_ = torch.sigmoid(torch.mm(a1, W2))

    loss = torch.mean(Y * torch.log(y_) + (1-Y) * torch.log(1 - y_))
    loss.backward()

    w1.data = w1.data - learning_rate * w1.grad.data
    w2.data = w2.data - learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()


''' 
    + Torch's nn package
        Provide's keras like implementation and optimization.

        torch.nn.Sequential(
            Layers ...
        )

        - Functional : torch.nn.functional (import as F) provides all the activation functions (and maybe other helpful functions).
            F.relu(torch.nn.Linear(in_dim, output_dim))
            F.softmax()
            F.sigmoid()
            F.tanh()

        - Layers 
            torch.nn.Linear(input_dim, output_dim)
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

            torch.nn.ReLU()

        - Losses :
            torch.nn.MSELoss          - Mean Squared Error
            torch.nn.BCELoss          - Binary Cross Entropy
            torch.nn.KLDivLoss        - KL Divergence
            torch.nn.CrossEntropyLoss - Softmax Cross Entropy



        - Optimizers : Instead of updating the weight's ourselves, we can use pre defined optimizers.

            model = torch.nn.Sequential(...)
            loss = torch.nn.MSELoss(size_average=False)
            optim = torch..optim.AdamOptimizer(model.parameters(), lr=learning_rate)

            for _ in range(no_steps):
                y_pred = mosel(x)
                lo = loss(y_pred, y)
                optim.zero_grad()
                lo.backward()
                optim.step()

    This can be achieved using an optimizer and without.
'''

model = torch.nn.Sequential(
        torch.nn.Linear(50, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )

loss = torch.nn.MSELoss(size_average=False) # use this or Loss2
for _ in range(1000):
    y_pred = model(X)
    
    loss2 = torch.mean(torch.squrare(y_pred - Y))

    loss.backward()

    for i in model.parameters():
        i.data.sub_(learning_rate * i.grad)
        i.grad.data.zero_()

model = torch.nn.Sequential(
        torch.nn.Linear(50, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )


loss = torch.nn.MSELoss(size_average=False)
optim = torch..optim.AdamOptimizer(model.parameters(), lr=learning_rate)

for _ in range(no_steps):
    y_pred = model(X)
    lo = loss(y_pred, Y)
    optim.zero_grad()

    lo.backward()
    optim.step()


'''
    + Custom Models
        Custom models are way to go! Subclass torch.nn.Module and define all the layers in constructor &
        define the forward function which calls all of them.
    
'''


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(TwoLayerNet, self).__init__()
        self.lin1 = torch.nn.Linear(D_in, 20)
        self.lin2 = torch.nn.Linear(20, D_out)

    def forward(self, x):
        a1 = self.lin1(x).clamp(min=0)
        a2 = self.lin2(a1)
        return a2

X = Variable(torch.nn.randn(1000, 50), requires_grad=False)
Y = Variable(torch.nn.randn(1000, 1), requires_grad=False)

model = TwoLayerNet(50, 1)
loss = torch.nn.MSELoss(size_average=False)
optim = torch.optim.AdamOptimizer(model.parameters(), lr=1e-4)

for _ in range(no_steps):
    y_pred = model(X)
    lola = loss(y_pred, Y)

    lola.backward()
    optim.step()
    optim.zero_grad()

'''
    + CNN
        - torch.nn.Conv2d(input_channels=3, out_channels/no_filters, filter_size (tuple), strides (tuple), padding=0/1)
        - torch.nn.MaxPool2d(kernel_size, stride=None, padding=0)
'''

class ConvNet(torch.nn.Module):
    def __init__(self, in_cha, out_cha):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_cha, 64, (9, 9), (1, 1))
        self.conv2 = torch.nn.Conv2d(64, 32, (7, 7), (1, 1))
        self.conv3 = torch.nn.Conv2d(32, out_cha, (5, 5), (1, 1))
        self.act = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(5)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x2 = self.max_pool(x2)
        x3 = self.conv3(x2)
        x3 = self.max_pool(x3)
        return x3

model = ConvNet(3, 3)
model(X)

''' 
    + Word Embeddings 
        - Create a dictionary with the keys being the words, and the values being the indexes.
            w2ix = {"hello" : 0, "world" : 1}
            w2ix = { word: i for i, word in enumerate(vocab)}
    
        - Define embedding with vocab size and embedding_dim.
            embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        - Create a tensor for indices and pass it to the embedding
            for sent in sentences :
                input = torch.Tensor([w2ix[word] for word in sent.split(" ")], torch.long)
                embed = embedding(input)

        - CONTEXT_SIZE : No. of words passed to input 
            Normally, you don't pass the whole sentence, but a few words. 
            The next LINEAR layer will have embedding * context as input_dim as you reshape it to that dim.

'''

class EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(EmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = torch.nn.Linear(embedding_dim * context_size, 100)
        self.lin2 = torch.nn.Linear(100, vocab_size) # IMP : Since we are predicting next word, return one-hot of vocab_size.

    def forward(self, input):
        embed = self.embedding(input).view(1, -1) # IMP : .view() = .reshape()
        out = F.relu(self.lin1(embed))
        out = F.softmax(self.lin2(out))
        return out


text = ''' When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days; '''

sentences, words = text.split("\n"), text.split()
vocab = list(set([word for word in words]))

vocab_size, epoch, embedding_dim, learning_rate = len(vocab), 10, 128, 0.001
context_size = 3 # pass 3 words as input, predict next one.

word2index = {word : i for i, word in enumerate(vocab)}

dataset, j = list(), 0
for i in range(0, len(words), 3):
    dataset[j] = [[words[i], words[i+1]], words[i+2]]
    j += 1

model = EmbeddingModel(vocab_size, embedding_dim, context_size)
loss = torch.nn.NLLLoss()
optim = torch.optim.AdamOptimizer(model.parameters(), lr=learning_rate)

for _ in range(epoch):
    for context, target in dataset:
        inp = torch.LongTensor([word2index[c] for c in context])

        model.zero_grad()
        output = model(inp)
        l_val = loss(output, torch.LongTensor(word2index[target])) # IMP
        l_val.backward()
        optim.step()


'''
    + LSTMs
        - Create an LSTM object with input & output dimensions & the input has to be a list of Tensors.
          The input sequence is later reshaped to hidden_state dim.
            
            lstm = torch.nn.LSTM(embedding_dim, no_classes)
            inputs = [torch.randn(1, embedding_dim) for _ in sequence_len]

        - Define number of hidden states (stacked on top), 
          w/ dimensions : [no_sequence, mini_batch=1, embedding_dim]

            hidden = (torch.randn(1, 1, embedding_dim),
                      torch.randn(1, 1, embedding_dim))

        - You can get output for every input one at a time.
            
            for i in inputs:
                out, hidden = lstm(i.view(1, 1, -1), hidden)

'''

lstm = torch.nn.LSTM(embedding_dim, 10)
inputs = [torch.randn(1, embedding_dim) for _ in sequence_len]

hidden = (torch.randn(1, 1, embedding_dim), 
         torch.randn(1, 1, embedding_dim))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)


'''
    TODO :
    ModuleDict
    
'''

