# import read_input
# import output
import torch
import torchvision as vision

class model():
    def __init__(self,model = 'vgg16'):
        if model=='vgg16':
            self.cv_model = vision.models.vgg16(pretrained = True)
        elif model == 'resnet101':
            self.cv_model = vision.models.resnet101(pretrained= True)
        elif model == 'resnet152':
            self.cv_model = vision.models.resnet152(pretrained=True)
        elif model == 'squeezenet':
            self.cv_model = vision.models.squeezenet1_0(pretrained = True)
        elif model == 'vgg11':
            self.cv_model = vision.models.vgg11(pretrained=True)
        elif model == 'vgg13':
            self.cv_model = vision.models.vgg13(pretrained=True)
        elif model == 'vgg16':
            self.cv_model = vision.models.vgg16(pretrained=True)
        elif model == 'resnet18':
            self.cv_model = vision.models.resnet18(pretrained=True)
        elif model == 'resnet34':
            self.cv_model = vision.models.resnet34(pretrained=True)
        elif model == 'resnet50':
            self.cv_model = vision.models.resnet50(pretrained=True)
        else:
            self.cv_model = vision.models.resnet18(pretrained=True)
        
    def init_optimizer(self,optimizer,learning_rate):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.cv_model.parameters(),lr = learning_rate)           
        elif optimizer == 'AdaGrad':
            self.optimizer = torch.optim.Adagrad(self.cv_model.parameters(),lr = learning_rate)           
        elif optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.cv_model.parameters(),lr = learning_rate)           
        elif optimizer == 'ASGD':
            self.optimizer = torch.optim.ASGD(self.cv_model.parameters(),lr = learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.cv_model.parameters(),lr= learning_rate)
        elif optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.cv_model.parameters(),lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.cv_model.parameters(),lr=learning_rate)
    
    def init_loss(self,output,label,loss):
        if loss == 'MSE':
            self.loss= torch.nn.MSELoss(output,label)
        elif loss == 'L1':
            self.loss = torch.nn.L1Loss(output,label)
        elif loss == 'crossentropy':
            self.loss = torch.nn.CrossEntropyLoss(output,label)
        elif loss == 'softmargin':
            self.loss = torch.nn.SoftMarginLoss(output,label)
        else:
            file = open('/home/ubuntu/logfile.txt','w+')
            file.write("no loss choosed, quit the program.")
            quit()
        return self.loss

    def train(self,input,label,epoch,learning_rate=1e-4,optimizer='Adam'):
        """ Normally used for fine tuning of networks. 
        
        Arguments:
            input: the input data. Dimension: N * Height* Width * 3
            label: the labeled class. Dimension: N * 2000 (one hot)
            epoch: steps to run
            learning_rate: learning rate. Default is 1e-4 because it's fine tuning.
            optimizer: optimizer. Default is Adam

        Returns:
            None
        """
        self.init_optimizer(optimizer,learning_rate)
        for i in range(epoch):
            output = self.cv_model(input)
            loss = self.init_loss(output,label,'MSE')
            optimizer = self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        return None

    def prediction(self,input,output_location):
        """ make runtime prediction based on pretrained model

        """ 
        output = self.cv_model(input)

        return None