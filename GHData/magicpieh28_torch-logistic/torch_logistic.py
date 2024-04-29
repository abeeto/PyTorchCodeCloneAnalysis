import torch
import torch.nn as nn
import torch.optim as optim
from torch_train import make_train_data
from torch_train import make_BoW
from collections import Counter
from pathlib import Path

# Test data
dir_folder = Path('/Users/jungwon-c/Documents/ML Logistic/data')
unlabeled = dir_folder/'book'/'unlabeled.review'


counter = Counter()

# batch
def iteration(datum, target, batch_size: int):
	for i in range(0, len(datum), batch_size):
		yield datum[i : i + batch_size], target[i : i + batch_size]

# logistic regression
class LogisticR(nn.Module):
	def __init__(self, vocab_size, label_size):
		super().__init__()
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.linear = nn.Linear(self.vocab_size, self.label_size)

	def forward(self, BoWvec):
		# UserWarning: nn.functional.sigmoid is deprecated.
		# Use torch.sigmoid instead.warnings.
		# warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
		y_pred = torch.sigmoid(self.linear(BoWvec))
		return y_pred

model = LogisticR(vocab_size=1000, label_size=2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

datum, target = make_train_data(1000, counter)
# Expected object of type torch.LongTensor
# but found type torch.FloatTensorfor argument
datum = torch.tensor(datum, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)

model.train()

for epoch in range(100):
	print(f'-- {epoch+1} --')
	# 왠지 모르겠지만 첫 epoch 때에만 loss가 많이 출력됨
	for datum, target in iteration(datum, target, 8):
		model.zero_grad()

		y_pred = model(datum)
		loss = loss_func(y_pred, target)

		loss.backward()
		optimizer.step()

		print(f'loss => {loss}')

with unlabeled.open(mode='r', encoding='utf-8') as unla:
	with torch.no_grad():
		input_datum = []
		for sentence in unla:
			input_datum.append(make_BoW(300, sentence))
		print(input_datum)
		pred = model(input_datum[0])
		print(pred)