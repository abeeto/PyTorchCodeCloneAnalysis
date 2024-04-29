import torchtext.datasets as datasets
from torchtext import data 
TEXT = data.Field(lower = True,batch_first = True,
	fix_length = 20)
LABEL = data.Field(sequential = False)
train,test = datasets.IMDB.splits(TEXT,LABEL)

#Build the vocabulary 
TEXT.build_vocab(train,vectors = GloVe(name = '6B',
	dim = 300),max_size = 10000,min_freq = 10)
LABEL.build_vocab(train)

print(TEXT.vocab.freqs)

train_iter,test_iter = data.BucketIterator.splits((train,test),
	batch_size = 128,device = -1,shuffle = True)
batch = next(iter(train_iter))
batch.text 

class EmbNet(nn.Module):
	def __init__(self,emb_size,hidden_size1,hidden_size2 = 400):
		super().__init__()
		self.embedding = nn.Embedding(emb_size,hidden_size1)
		self.fc = nn.Linear(hidden_size2,3)

	def forward(self,x):
		embeds = self.embedding(x).view(x.size(0),-1)
		out = self.fc(embeds)
		return F.log_softmax(out,dim = -1)


def fit(epoch,model,data_loader,phase = 'training',volatile = False):
	if phase == 'training':
		model.train() 
	if phase == 'validation':
		model.eval()
		volatile = True 
	running_loss = 0.0
	running_correct = 0
	for batch_idx,batch in enumerate(data_loader):
		text,target = batch.text,batch.label 
		if is_cuda:
			text,target = text.cuda(),target.cuda()
		if phase == 'training':
			optimizer.zero_grad()
		output = model(text)
		loss = F.nll_loss(output,target)
		running_loss += F.nll_loss(output,target,size_average = False).data[0]
		preds = output.data.max(dim = 1,keepdim = True)[0]
		running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
		if phase == 'training':
			loss.backward()
			optimizer.step()
	loss = running_loss/len(data_loader.dataset)
	accuracy = 100.*running_correct/len(data_loader.dataset)
	return loss,accuracy

train_losses,train_accuracy = [],[]
val_losses,val_accuracy = [],[]
train_iter.repeat = False 
test_iter.repeat = False 

for epoch in range(1,10):
	epoch_loss,epoch_accuracy = fit(epoch,model,train_iter,phase = 'training')
	val_epoch_loss,val_epoch_accuracy = fit(epoch,model,test_iter,phase = 'validation')
	train_losses.append(epoch_loss)
	train_accuracy.append(epoch_accuracy)
	val_losses.append(val_epoch_loss)
	val_accuracy.append(val_epoch_accuracy)
	

