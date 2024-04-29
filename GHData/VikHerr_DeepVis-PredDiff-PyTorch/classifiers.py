# Classifier functions

import torch 
import torchvision as tv

from config import CLASSES

class Classifier:

    def __init__(self, model, softmax=True, categroies="model/imagenet_classes.txt"):
        '''
        categroies: path to result categories
        ood : function for post processing
        '''

        self.name = model['name']
        # self.cPath = model['path'] # only for costum models
        self.gpu = torch.cuda.is_available()
        self.model = self.__getClassifier(model)
        self.categories = self.__getCategories(categroies)
        self.softmax    = softmax
        self.ood        = model['ood']

    def predict(self,x):

        probabilities = self.forward_pass(x)
        
        probabilities = probabilities.cpu().detach().numpy()

        #cat_prob = self.getCatProb(probabilities,category_idx)

        return probabilities



    def forward_pass(self, x):
        '''
        forward pass of pytorch net
        '''
        if self.gpu:
            x = x.to('cuda')

        with torch.no_grad():
            output = self.model(x)

        if self.ood:
            probabilities = torch.nn.functional.softmax(output, dim=1)
            return self.ood(probabilities,CLASSES)
        elif self.softmax:
            probabilities = torch.nn.functional.softmax(output, dim=1)

        else:
            return output

        return probabilities

    def getCatProb(self,prob,cat_idx):
        '''
        get probability of specific categorie
        '''

        return prob[:,cat_idx]
          
    def getOod(self,x):
        y = self.predict(x)
        return y


    def getTopCats(self,x, top=1):
        '''
        get top categorys
        '''
        temp = self.softmax
        # ensure softmax here
        self.softmax = True
        if self.gpu:
            x = x.to('cuda')

        prob = self.forward_pass(x)
        self.softmax = temp

        return torch.topk(prob, top)
    
    def getTopCat(self,x):
        '''
        get idx of top category
        '''
        if self.ood:
            return self.getOod(x), int(0)
        else:
            top_probs, top_catids = self.getTopCats(x,top=1)
            return top_probs[0][0].item(), top_catids[0][0].item()
        
    def report(self):
        print('eval model: \n', self.model)

    def __getClassifier(self, modelInfo):
        '''
        define classifier model
        '''
        model = modelInfo['model']
        model.eval()

        if self.gpu:
            model.to('cuda')

        return model


    def __getCategories(self, path):
        '''
        load categories for classifier
        '''
        with open(path, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        return categories



        

        