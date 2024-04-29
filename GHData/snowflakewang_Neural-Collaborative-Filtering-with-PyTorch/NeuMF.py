'''
Created on Sep 4, 2021
PyTorch Implementation of Multi-Layer Perceptron recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Yitong Wang (yitongwang.snowflake@gmail.com)
'''

import numpy as np
import torch
from torch import nn,optim
from time import time
import sys
import argparse
import multiprocessing as mp
from Dataset import Dataset
from evaluate import evaluate_model

###################Arguments###################
def parse_args():
    parser=argparse.ArgumentParser(description='Multi-Layer Perceptron')
    parser.add_argument('--path',nargs='?',default='D:/我的学习/USTC/数据科学实验室(Lab for Data Science)/Neural Collaborative Filtering/Data/',help='input data path')#路径需要根据实际情况修改
    parser.add_argument('--dataset',nargs='?',default='',help='choose a dataset')#数据集名称需要根据实际情况修改
    parser.add_argument('--epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size')
    parser.add_argument('--num_factors',type=int,default=8,help='embedding size of MF model')
    parser.add_argument('--layers',nargs='?',default='[64,32,16,8]',help='size of each layer. note that the first layer'
                                                                         'is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size')
    parser.add_argument('--reg_mf',type=float,default=0,help='regularization for MF embeddings')
    parser.add_argument('--reg_layers',nargs='?',default='[0,0,0,0]',help='regularization for each MLP layer. reg_layers[0] is the regularization for embeddings')
    parser.add_argument('--num_neg',type=int,default=4,help='number of negative instances to pair with a positive instance')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--learner',nargs='?',default='adam',help='specify an optimizer: adam, adagrad, rmsprop, sgd')
    parser.add_argument('--verbose',type=int,default=1,help='show performance per X iterations')
    parser.add_argument('--out',type=int,default=1,help='whether to save the model')
    parser.add_argument('--mf_pretrain',nargs='?',default='',help='specify the pretrain model file for MF part. if empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain',nargs='?',default='',help='specify the pretrain model file for MLP part. if empty, no pretrain will be used')
    return parser.parse_args()

class NeuMF(nn.Module):
    def __init__(self,num_users,num_items,mf_dim=10,layers=[10],reg_layers=[0],reg_mf=0):
        super(NeuMF, self).__init__()
        assert len(layers)==len(reg_layers)
        self.num_layer=len(layers)
        self.num_users=num_users
        self.num_items=num_items
        self.mf_dim=mf_dim
        self.layers=layers
        self.reg_layers=reg_layers
        self.reg_mf=reg_mf

        self.GMF_user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.GMF_item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        self.GMF_user_latent = nn.Flatten()
        self.GMF_item_latent = nn.Flatten()

        self.MLP_user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        self.MLP_user_latent = nn.Flatten()
        self.MLP_item_latent = nn.Flatten()
        if self.num_layer>1:
            mlp = []
            for i in range(self.num_layer):
                if i!=self.num_layer-1:
                    mlp.append(nn.Linear(in_features=self.layers[i], out_features=self.layers[i + 1]))
                    mlp.append(nn.ReLU())
            self.mlp=nn.Sequential(*mlp)
        neumf=[]
        neumf.append(nn.Linear(in_features=self.layers[-1]+mf_dim,out_features=1))
        neumf.append(nn.Sigmoid())
        self.neumf=nn.Sequential(*neumf)

    def forward(self,user_x,item_x):
        GMF_user_embedding = self.GMF_user_embedding(user_x)
        GMF_item_embedding = self.GMF_item_embedding(item_x)
        GMF_user_latent = self.GMF_user_latent(GMF_user_embedding)
        GMF_item_latent = self.GMF_item_latent(GMF_item_embedding)
        GMF_vector=torch.mul(GMF_user_latent,GMF_item_latent)

        MLP_user_embedding = self.MLP_user_embedding(user_x)
        MLP_item_embedding = self.MLP_item_embedding(item_x)
        MLP_user_latent = self.MLP_user_latent(MLP_user_embedding)
        MLP_item_latent = self.MLP_item_latent(MLP_item_embedding)
        MLP_user_item_concat = torch.cat((MLP_user_latent, MLP_item_latent), dim=1)  # 是按行拼接一上一下？
        MLP_vector=self.mlp(MLP_user_item_concat)

        NeuMF_vector=torch.cat((GMF_vector,MLP_vector),dim=1)
        prediction=self.neumf(NeuMF_vector)
        return prediction

def get_model(num_users,num_items,mf_dim=10,layers=[10],reg_layers=[0],reg_mf=0):
    return NeuMF(num_users,num_items,mf_dim,layers,reg_layers,reg_mf)

def load_pretrain_model(model,gmf_model,mlp_model,num_layers):
    pass

def get_train_instances(train,num_negatives):
    user_input,item_input,labels=[],[],[]
    num_users=train.shape[0]
    for (u,i) in train.keys():
        #positive instances
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        #negative instances
        for t in range(num_negatives):
            j=np.random.randint(num_items)
            while train.__contains__((u,j)):
                j=np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input,item_input,labels

if __name__=='__main__':
    args=parse_args()
    num_epochs=args.epochs
    batch_size = args.batch_size
    mf_dim=args.num_factors
    layers=eval(args.layers)
    reg_mf=args.reg_mf
    reg_layers=eval(args.reg_layers)
    num_negatives=args.num_neg
    learner=args.learner
    learning_rate=args.lr
    verbose=args.verbose
    mf_pretrain=args.mf_pretrain
    mlp_pretrain=args.mlp_pretrain

    top_K=10
    evaluation_threads=1#mp.cpu_count()
    print("NeuMF arguments: %s"%(args))
    model_out_file='pretrain/%s_NeuMF_%s_%d.h5'%(args.dataset,args.layers,time())

    #loading data
    t1=time()
    dataset=Dataset(args.path+args.dataset)
    train,testRatings,testNegatives=dataset.trainMatrix,dataset.testRatings,dataset.testNegatives
    num_users,num_items=train.shape
    print("Load data done [%.1f s]. #user=%d, item=%d, #train=%d, #test=%d"%(time()-t1,num_users,num_items,train.nnz,len(testRatings)))

    model=get_model(num_users,num_items,mf_dim,layers,reg_layers,reg_mf)
    criterion = nn.BCELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    '''
    if learner.lower() == "adagrad":
        optimizer = optim.Adagrad
    elif learner.lower() == "rmsprop":
        optimizer = optim.RMSprop
    elif learner.lower() == "adam":
        optimizer = optim.Adam
    else:
        optimizer = optim.SGD
    '''

    #initial performance
    (hits,ndcgs)=evaluate_model(model,testRatings,testNegatives,top_K,evaluation_threads)
    hr,ndcg=np.array(hits).mean(),np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f'%(hr,ndcg))
    best_hr,best_ndcg,best_iter=hr,ndcg,-1
    '''
    if args.out>0:
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(state,model_out_file)
    '''

    #training model
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    for epoch in range(num_epochs):
        num_batch,epoch_loss=0,0.0
        t1=time()
        #Generate training instances
        user_input,item_input,labels=get_train_instances(train,num_negatives)
        user_input=torch.tensor(user_input)
        item_input=torch.tensor(item_input)
        labels=torch.tensor(labels,dtype=torch.float)
        training_dataset=torch.utils.data.TensorDataset(user_input,item_input,labels)
        train_iter=torch.utils.data.DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
        for i,batch in enumerate(train_iter):
            user_x,item_x,label_x=batch[0],batch[1],batch[2]
            user_x=user_x.to(device)
            item_x=item_x.to(device)
            label_x=label_x.to(device)
            print('user_x: ',user_x)
            print('item_x: ',item_x)
            print('label_x: ',label_x)
            num_batch+=len(user_x)
            optimizer.zero_grad()
            score=model(user_x,item_x)
            print('the shape of score is: ',score.shape)
            loss=criterion(score.t()[0],label_x)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        t2=time()

        if epoch%verbose==0:
            (hits,ndcgs)=evaluate_model(model,testRatings,testNegatives,top_K,evaluation_threads)
            hr,ndcg=np.array(hits).mean(),np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, epoch_loss, time() - t2))
            if hr > best_hr:  # 保存best结果
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    '''
    if args.out > 0:
        print("The best NeuMF model is saved to %s" % (model_out_file))
    '''