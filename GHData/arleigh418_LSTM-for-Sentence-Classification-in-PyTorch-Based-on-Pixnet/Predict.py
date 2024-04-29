import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random
from LSTM_sentence_classifier import LSTMClassifier
import pandas as pd



predict_data,word_to_ix = data_loader.predict_data("test2.xlsx") 

model = LSTMClassifier(400, 400, len(word_to_ix),8)
model.load_state_dict(torch.load('best_model_acc7867.model',map_location='cpu')) #map_location='cpu': whether you use gpu or not



def predict(model, data,word_to_ix):
    print("start evaluate")
    model.eval()
    unknown_error =0
    pred_res = []
    replace_label = []
    # print(data)
    for predict_sent in data:
        
        model.hidden = model.init_hidden()
        predict_sent = data_loader.prepare_predict_sequence(predict_sent, word_to_ix)
        try:
            pred = model(predict_sent)
            
        except:
            pred = torch.rand(1, 8)
            unknown_error=unknown_error+1
        print('無法辨識:',unknown_error)
        # print(pred)
        pred_label = pred.data.max(1)[1].numpy()
        
        pred_res.append(pred_label)
    for x in pred_res:
        # print(x)
        if x == 0:
            replace_label.append('星座')
        elif x == 1:
            replace_label.append('美食')
        elif x == 2:
            replace_label.append('美妝')
        elif x == 3:
            replace_label.append('醫療')
        elif x == 4:
            replace_label.append('親子')
        elif x == 5:
            replace_label.append('電影')
        elif x == 6:
            replace_label.append('運動體育')
        elif x == 7:
            replace_label.append('旅遊')
        
    data_store = pd.DataFrame(replace_label)
    data_store.to_excel('predict_testuse.xlsx')


   

predict(model,predict_data,word_to_ix)

