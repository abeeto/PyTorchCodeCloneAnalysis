from models import SkipGramDataset
from torch.utils.data import DataLoader
from models import EmbeddingNN, SkipGram
import pickle
from torch.optim import SGD

def train():
  dataset = SkipGramDataset('train_pairs.p')
  word2idx, idx2word = pickle.load(open('dictionary.p', 'rb'))

  batch_size = 1
  epochs = 10
  voc_size = len(word2idx)
  total_batches = len(dataset) // batch_size

  embeddings = EmbeddingNN(voc_size, 300)
  skipgram = SkipGram(embeddings)
  opt = SGD(skipgram.parameters(), lr=1)

  for e in range(1, 1+epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_id, data in enumerate(dataloader):
      loss = skipgram(data)
      opt.zero_grad()
      loss.backward()
      opt.step()
      if (batch_id + 1) % 10 == 0:
        print('epoch {} : batch {} / {} loss: {}'.format(
          e, batch_id+1, total_batches, loss.data[0]))

if __name__ == "__main__":
  train()
