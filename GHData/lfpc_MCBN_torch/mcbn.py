import torch
from copy import copy
import utils

class MCBN_Ensemble(torch.nn.Module):
    def __init__(self,model,
                    n_samples:int, #number of MonteCarlo samples
                    batch_loader,#DataLoader to sample random mini-batches (e.g. training dataloader)
                    return_average:bool = True #If True, returns average of ensemble samples as output. If false, return all samples
                    ):
        super().__init__()
        
        assert isinstance(batch_loader.sampler,torch.utils.data.sampler.RandomSampler), "Batch Loader should have shuffle set to True to give randomness"
        self.batch_loader = batch_loader
        self.model = model
        self.n_samples = n_samples
        self.return_average = return_average
        self.p = torch.nn.Parameter(torch.tensor(0.5,requires_grad = True)) #dummy parameter

        self.__get_BN_modules()
        self.__save_main_attributes()
        
    def to(self,device):
        super().to(device)
        self.model.to(device)
        return self

    def __get_BN_modules(self):
        '''Store model modules respetive to BatchNormalization'''
        self.__modules = []
        for m in self.model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                self.__modules.append(m)
        assert len(self.__modules)>0, "No BatchNormalization modules found in model"

    def __save_main_attributes(self):
        '''Store original BatchNormalization attributes so after MCBN evaluations
        the model can be reseted to it's original evaluation mode'''
        self.__momentum = {}
        self.__running_mean= {}
        self.__running_var= {}

        for m in self.__modules:
            self.__momentum[m] = copy(m.momentum)
            self.__running_mean[m] = copy(m.running_mean)
            self.__running_var[m] = copy(m.running_var)


    def __set_main_attributes(self):
        '''Reset BatchNormalization original (inference mode) attributes'''
        for m in self.__modules:
            m.momentum = copy(self.__momentum[m])
            m.running_mean = copy(self.__running_mean[m])
            m.running_var = copy(self.__running_var[m])

    def set_BN_mode(self):
        '''Turn BN to training mode:
           The mean and variance are calculated taking into account only the present mini-batch.'''
        for m in self.__modules:
            m.train()
            m.track_running_stats = True #Set BN modules to train mode where the running means and vars are updated
            m.momentum = 1 #Set momentum to 1, so running means and vars don't take old values into consideration, only the present batch information

            
    def reset_normal_mode(self):
        '''Set model to eval mode again, with Batch Normalization turned off'''
        self.eval()
        self.__set_main_attributes()


    def get_samples(self,x):
        '''Sample from random models (random BN means and vars) and store all outputs'''
        ensemble = []
        batch_loader = iter(self.batch_loader)
        for _ in range(self.n_samples):
            im_train,_ = next(batch_loader)
            im_train = im_train.to(self.device)
            self.set_BN_mode()
            with torch.no_grad():
                self.model(im_train) #Inference in random batch with BN mode setted, so running statistics are updated
                self.model.eval() #Set eval mode so the model doesn't update running statistics (use the ones calculated in the previous line)
                y = self.model(x)
                ensemble.append(y)
                self.ensemble = torch.stack(ensemble)
        return self.ensemble

        
    def forward(self,x):
        y = self.get_samples(x)

        if self.return_average:
            '''Output as average of ensemble samples'''
            y = torch.mean(self.ensemble,axis = 0)
        return y


def set_BN_mode(model):
    '''Turn BN to training mode:
    The mean and variance are calculated taking into account only the present mini-batch.'''
    for m in model.modules():
        if m.__class__.__name__.startswith('BatchNorm'):
            m.train()
            m.track_running_stats = True #Set BN modules to train mode where the running means and vars are updated
            m.momentum = 1 #Set momentum to 1, so running means and vars don't take old values into consideration, only the present batch information
def store_main_attributes(model):
        '''Store original BatchNormalization attributes so after MCBN evaluations
        the model can be reseted to it's original evaluation mode'''
        momentum = {}
        running_mean= {}
        running_var= {}

        for m in model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                momentum[m] = copy(m.momentum)
                running_mean[m] = copy(m.running_mean)
                running_var[m] = copy(m.running_var)
        return running_mean, running_var, momentum

        
def reset_normal_mode(model,**kwargs):
    '''Set model to eval mode again, with Batch Normalization turned off'''
    model.eval()
    '''Reset BatchNormalization original (inference mode) attributes'''
    for m in model.modules():
        if m.__class__.__name__.startswith('BatchNorm'):
            for attribute,value in kwargs.items():
                if attribute == 'running_mean': #don't listed in module.__dict__
                    m.running_mean = value[m]
                elif attribute == 'running_var': #don't listed in module.__dict__
                    m.running_var = value[m]
                else:
                    m.__dict__[attribute] = copy(value[m])
    return model

def get_samples(model,x,batch_loader, n_samples:int):
        '''Sample from random models (random BN means and vars) and store all outputs'''
        device = next(model.parameters()).device
        ensemble = []
        batch_loader = iter(batch_loader)
        for _ in range(n_samples):
            im_train,_ = next(batch_loader)
            im_train = im_train.to(device)
            set_BN_mode(model)
            with torch.no_grad():
                model(im_train) #Inference in random batch with BN mode setted, so running statistics are updated
                model.eval() #Set eval mode so the model doesn't update running statistics (use the ones calculated in the previous line)
                y = model(x)
                ensemble.append(y)
                ensemble = torch.stack(ensemble)
        return ensemble