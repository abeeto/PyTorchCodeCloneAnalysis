# ______________________________________ IMPORT PACKAGES ___________________________________

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
pd.core.common.is_list_like=pd.api.types.is_list_like
import pandas_datareader.data as web
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

style.use('ggplot')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ticker = 'GOOGL'

# In[ ]: DATA COLLECTION

# Pull Data into CSV Files

def get_data_from_iex(reload_sp500=False):
    
    
    start = dt.datetime(2013,1,1)
    end = dt.datetime(2018,12,31)    
    df = web.DataReader(ticker, 'iex', start, end)
    df.to_csv('stock_dfs/{}.csv'.format(ticker))
    

get_data_from_iex()

# Combine all ticker closing data into one CSV

def compile_data():

    main_df = pd.DataFrame()  
    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
    df['date']=pd.to_datetime(df.date)
    df['date']=df['date'].dt.strftime('%Y%m%d')
    df.set_index('date', inplace=True)
    df.rename(columns = {'close':ticker}, inplace=True)
    df.drop(['open','high','low','volume'], 1, inplace=True)
    main_df =df
        
    print(main_df.head())
    main_df.to_csv('googl_closes.csv')

compile_data()

# In[ ]: ENCODER / DECODER

class encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, T):
       
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)

    def forward(self, input_data):  
        
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())        
        hidden = self.init_hidden(input_data) 
        cell = self.init_hidden(input_data)
        
        for t in range(self.T - 1):            
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2)             
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)) 
            attn_weights = F.softmax(x.view(-1, self.input_size),dim=1)             
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])             
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]            
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
            
        return input_weighted, input_encoded

    def init_hidden(self, x):
        
        return Variable(x.data.new(1, x.size(0), self.hidden_size, device=device).zero_()) 

           
class decoder(nn.Module):
    
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        
        super(decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size        
        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)
        self.fc.weight.data.normal_()

        
    def forward(self, input_encoded, y_history):
        
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
       
        for t in range(self.T - 1):          
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, self.T - 1),dim=1)             
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            
            if t < self.T - 1:                
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1))                 
                self.lstm_layer.flatten_parameters()                
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0] 
                cell = lstm_output[1] 
        
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))*1000
        return y_pred
    
    def init_hidden(self, x):
        
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size, device=device).zero_())

# In[ ]: DUAL-STAGE ATTENTION-BASED RECURRENT NEURAL NETWORK

class da_rnn:
    
    def __init__(self, file_data, encoder_hidden_size = 65, decoder_hidden_size = 65, T = 10,
                 learning_rate = 0.01, batch_size = 130, parallel = True, debug = False):
        
        self.T = T
        dat = pd.read_csv(file_data, nrows = 100 if debug else None)        
        self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'GOOGL']].values    
        self.y = np.array(dat.GOOGL)
        self.y.resize(len(self.y),1)       
        self.batch_size = batch_size
        self.encoder = encoder(input_size = self.X.shape[1], hidden_size = encoder_hidden_size, T = T).cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size, decoder_hidden_size = decoder_hidden_size, T = T).cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)   
        self.train_size = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[:self.train_size])      

    def train(self, n_epochs = 10):
        
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))        
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)
        self.loss_func = nn.MSELoss()
        n_iter = 0
               
        
        for i in range(n_epochs):            
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            
            while j < self.train_size:                
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):                    
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T - 1), :]
                    g = self.y[batch_idx[k] : (batch_idx[k] + self.T - 1)]
                    g.resize(1,9)
                    y_history[k] = g                              
                
                loss = self.train_iteration(X, y_history, y_target)
                m = int(i * iter_per_epoch + j / self.batch_size)
                self.iter_losses[m] = loss
                u=j / self.batch_size
                
                if (j / self.batch_size) % 50 == 0:
                    print(f"Epoch {i}, Batch {u}: Loss = {loss}")
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                            
            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {self.epoch_losses[i]}")                

            if i % 10 == 0:
                y_train_pred = self.predict(on_train = True)
                y_test_pred = self.predict(on_train = False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                
            if i == 9:
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y) + 1), y_test_pred, label = 'Test')
                plt.legend(loc = 'upper left')
                plt.title("Epoch = 10")
                plt.show()
                
            if i == 499:
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y) + 1), y_test_pred, label = 'Test')
                plt.legend(loc = 'upper left')
                plt.title("Epoch = 500")
                plt.show()         

    def train_iteration(self, X, y_history, y_target):
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_weighted, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
        y_pred = self.decoder(input_encoded, Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))
        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())
        loss = self.loss_func(y_pred, y_true)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.item()

    def predict(self, on_train = False):
        
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    h = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                    h.resize(1,9)
                    y_history[j] = h
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    s = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size - 1)]
                    s.resize(1,9)
                    y_history[j] = s
            
            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
            
        return y_pred
     
# In[ ]: MODEL

model = da_rnn(file_data = 'googl_closes.csv', parallel = False, learning_rate = .001)

model.train(n_epochs = 500)

y_pred = model.predict()


