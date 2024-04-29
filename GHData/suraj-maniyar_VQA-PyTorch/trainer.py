import torch
from torch.autograd import Variable
import pickle


def read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def train(config, model, train_loader, val_loader, optimizer, criterion):


    #TA, VA = [], []
    #TL, VL = [], []
 
    TA = read('results/train_accuracy.pkl')
    VA = read('results/val_accuracy.pkl')
    TL = read('results/train_loss.pkl')
    VL = read('results/val_loss.pkl')
 
    for epoch in range(config['epochs']):
    
        model.train()
        loss_train = 0
        total, correct = 0, 0
    
        for i, (x1, x2, y) in enumerate(train_loader):

            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                y = y.cuda()
       
            x1 = Variable(x1)
            x2 = Variable(x2)
            y = Variable(y)
        
            optimizer.zero_grad()
        
            outputs = model(x1, x2)
            train_loss = criterion(outputs, y)
        
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)

            if torch.cuda.is_available():
                correct += (y.cpu() == predicted.cpu()).sum(0)
            else:
                correct += (y == predicted).sum(0)
        
            loss_train += train_loss.item()*x1.size(0)
        
            train_loss.backward()
            optimizer.step()
        
  
            if i%100 == 0: 
                print('[%d/%d] \t Train Loss: %.4f \t Train Acc: %.4f \t %d/%d'  %  (i,
                                                                                len(train_loader),
                                                                                loss_train/(i+1),
                                                                                100.0*float(correct.item())/total,
                                                                                correct,
                                                                                total))
                torch.save(model, 'checkpoint/model_vgg16.pth')

        model.eval()
        loss_val = 0
        total_val, correct_val = 0, 0

        for i, (x1, x2, y) in enumerate(val_loader):

            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                y = y.cuda()

            x1 = Variable(x1)
            x2 = Variable(x2)
            y = Variable(y)


            outputs = model(x1, x2)
            val_loss = criterion(outputs, y)

            _, predicted = torch.max(outputs.data, 1)
            total_val += y.size(0)

            if torch.cuda.is_available():
                correct_val += (y.cpu() == predicted.cpu()).sum(0)
            else:
                correct_val += (y == predicted).sum(0)

            loss_val += val_loss.item()*x1.size(0)



            
        print('-'*120)
        print('Epoch %d \t Train Loss: %.3f \t Val Loss: %.3f \t Train Acc: %.3f \t Val Acc: %.3f'% ( epoch+1, 
                                                                                                      loss_train/len(train_loader),
                                                                                                      loss_val/len(val_loader),   
                                                                                                      100.0*correct.item()/total,
                                                                                                      100.0*correct_val.item()/total_val ))
        
        TL.append(loss_train/len(train_loader))
        VL.append(loss_val/len(val_loader))
        
        TA.append(100.0*correct.item()/total)
        VA.append(100.0*correct_val.item()/total_val) 

        with open('results/train_accuracy.pkl', 'wb') as f:
            pickle.dump(TA, f)
                
        with open('results/val_accuracy.pkl', 'wb') as f:
            pickle.dump(VA, f)

        with open('results/train_loss.pkl', 'wb') as f:
           pickle.dump(TL, f)

        with open('results/val_loss.pkl', 'wb') as f:
           pickle.dump(VL, f)

        print('-'*120)
        print('\n')



