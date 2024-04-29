'''
Created on Sep 4, 2021
PyTorch Implementation of Multi-Layer Perceptron recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Yitong Wang (yitongwang.snowflake@gmail.com)
'''

import scipy.sparse as sp#用来处理稀疏矩阵的科学计算库
import numpy as np

class Dataset(object):
    '''

    classdocs
    '''

    def __init__(self,path):
        '''

        constructor
        '''
        self.trainMatrix=self.load_rating_file_as_matrix(path+"ml-1m.train.rating")
        self.testRatings=self.load_rating_file_as_list(path+"ml-1m.test.rating")
        self.testNegatives=self.load_negative_file(path+"ml-1m.test.negative")
        assert len(self.testRatings)==len(self.testNegatives)

        self.num_users,self.num_items=self.trainMatrix.shape

    def load_rating_file_as_matrix(self,filename):
        '''

        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        #Get number of users and items
        num_users,num_items=0,0
        with open(filename,'r') as f:
            line=f.readline()
            while line != None and line != "":
                arr=line.split("\t")
                u,i=int(arr[0]),int(arr[1])
                num_users=max(num_users,u)
                num_items=max(num_items,i)
                line=f.readline()
        #Construct matrix
        mat=sp.dok_matrix((num_users+1,num_items+1),dtype=np.float32)
        with open(filename,'r') as f:
            line=f.readline()
            while line != None and line != "":
                arr=line.split("\t")
                user,item,rating=int(arr[0]),int(arr[1]),int(arr[2])
                if rating>0:
                    mat[user,item]=1.0
                line=f.readline()
        return mat

    def load_rating_file_as_list(self,filename):
        ratinglist=[]
        with open(filename,'r') as f:
            line=f.readline()
            while line != None and line != "":
                arr=line.split('\t')
                user,item=int(arr[0]),int(arr[1])
                ratinglist.append([user,item])
                line=f.readline()
        return ratinglist

    def load_negative_file(self,filename):
        negativelist=[]
        with open(filename,'r') as f:
            line=f.readline()
            while line != None and line != "":
                arr=line.split("\t")
                negatives=[]
                for x in arr[1:]:
                    negatives.append(int(x))
                negativelist.append(negatives)
                line=f.readline()
        return negativelist