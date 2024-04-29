import torch.nn as nn
import torch.nn.functional as F
class CNN1d(nn.Module):
    def __init__(self, embedding_model,n_filters, filter_sizes, output_dim, dropout):
        """
        Input:
            embedding model: fastText embedding model
            n_filters: integer
                number of filters
            filters_sizes: list
                list of filter size
            output_dim: int
                output dimension
            dropout: float value between [0,1)
        """
        super(CNN1d,self).__init__()
        self.embedding_model = embedding_model
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = 128, out_channels = n_filters, kernel_size = fs) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.lookup(self.embedding_model,text)   #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1) #embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs] #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1)) #cat = [batch size, n_filters * len(filter_sizes)]  
        return self.fc(cat)
    def lookup(self,embedding_model,x):
        """Uses fastText embedding model to get the vector representation of every words in the text in batch
        Input: 
            embedding_model: fastText embedding model
            x : batch_size X variable size sentence lenght
        Returns:
            retrun batch_size X maxlen X embedding_size numpy array
        """
        size = len(x) # size = batch_size
        maxlen =128

        embedded = np.empty(shape = (size,maxlen),dtype='U15') 
        
        for i in range(size):
            # intialize the ith row with <pad>
            embedded[i] = '<pad>' # taking minimum of lenght of sentence and max_len =128
            mini = min(len(x[i]),128) 
            if mini==128:# if minmum of the above two is 128 then truncate the previous part of the sentence
                embedded[i] = x[i][-128:] 
            else:# if minimum of the above two is length of sentence then using padding in the previous part of the sentence
                embedded[i][-mini:] = x[i]
        # creating a numpy array of batch_size X maxlen X embedding_size =128
        embedding = np.zeros(shape =(size,maxlen,128))
        
        for i in range(size):
            for j in range(maxlen):
                # using the embedding model to get the vector representation of the words in sentence
                embedding[i][j] = self.embedding_model[embedded[i][j]]
        return torch.Tensor(embedding)