import torch
import json
import os
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
from model import TriangleModel,clipped_sigmoid

from typing import List,Optional,Dict,Tuple,Callable,Union

BOUND = int(os.environ.get('BOUND',15))

### Clamp value for output
def invert_binary(tensor : torch.Tensor) -> torch.Tensor:
    '''
    Replace values of a binary tensor w/ +- BOUND
    
    Args:
        tensor (torch.Tensor) : binary values
    Returns:
        torch.Tensor with the same size as [tensor], but with
        all 1's set to BOUND and all 0's set to -BOUND.
    '''
    
    new_tensor = BOUND * torch.ones_like(tensor,device=tensor.device)
    new_tensor[tensor == 0] = -BOUND
    return new_tensor

def forward_euler(f : torch.nn.Module, x_0 : Dict[str,torch.Tensor],
                  t_0 : float, T : float, delta_t : float, detach = False) -> Dict[str,torch.Tensor]:
    '''
    Implementation of Forward Euler solver.
    
    Args:
        f (torch.nn.Module) : differential operator
        
        x_0 (Dict[str,torch.Tensor]) : Initial conditions
        
        t_0 (float) : Time at which to begin updating 
                      phonology and semantics
                      
        T (float) : Maxmimum time
        delta_t (float) : timestep
        
    Returns:
        Dictionary containing the values of all states / hidden units
        at each timestep.
    '''
    outputs,x = [x_0],x_0
    for t in torch.arange(0,T,delta_t):
        derivatives = f(x,detach=detach)
        nx = {}
        for key in x:
            if t<t_0 and key in ['phonology','semantics']:
               nx[key] = x_0[key]
               continue;
            else:
               nx[key] = x[key] + delta_t * derivatives[key]
               nx[key] = torch.clamp(nx[key],-BOUND,BOUND)
        outputs.append(nx)
        x = nx
    return outputs

class Metrics:
    def __init__(self,phoneme_embedding_matrix : torch.Tensor,
                      k : Optional[List[int]] = [1],
                      tau : Optional[List[float]] = [.5]):
        '''
        Utility class for phonological and semantic accuracies
        
        Args:
            phoneme_embedding_matrix (torch.Tensor) : feature vectors corresponding
                                                      to each phoneme
                                                      
            k (List[int]) : list of top-k phonological accuracies to compute
            tau (List[float]) : list of thresholds for computing semantic accuracy
        '''
        
        self.phoneme_embedding_matrix = phoneme_embedding_matrix
        self.k = k
        self.tau = tau
  
    def to(self, device : torch.device) -> 'Metrics':
        '''
        Copy [phoneme_embedding_matrix] to the provided device.
        
        Args:
            device (torch.device)
        Returns:
            self
        '''
        assert isinstance(device,torch.device)
        self.phoneme_embedding_matrix = self.phoneme_embedding_matrix.to(device)
        return self

    def compute_phon_accuracy(self,preds : torch.Tensor, targets : torch.Tensor,
                              k : int) -> float:
        '''
        Compute phonological accuracy
        
        Args:
            preds (torch.Tensor) : predicted phonology vectors
            targets (torch.Tensor) : ground-truth phonology vectors
            k (int) : setting for top-k accuracy
        Returns:
            Accuracy
        '''
        preds = preds.view(preds.shape[0],-1,1,self.phoneme_embedding_matrix.shape[-1])
        targets = targets.view(targets.shape[0],-1,1,self.phoneme_embedding_matrix.shape[-1])

        pred_distances = (preds - self.phoneme_embedding_matrix[None,None]).norm(dim=-1)
        target_distances = (targets - self.phoneme_embedding_matrix[None,None]).norm(dim=-1)

        vals = (target_distances.argmin(dim=-1,keepdim=True) == pred_distances.argsort(dim=-1)[:,:,:k]).any(dim=-1)
        return vals.all(dim=-1).float().mean().item()

    def compute_sem_accuracy(self,preds : torch.Tensor, targets : torch.Tensor,
                             tau : float) -> float:
        '''
        Compute semantic accuracy
        
        Args:
            preds (torch.Tensor) : predicted semantic vectors
            targets (torch.Tensor) : ground-truth semantic vectors
            tau (float) : threshold for binarizing [preds]
        Return:
            Accuracy
        '''
        acc = ((preds>=tau) == targets.bool()).all(dim=-1).float()
        return acc.mean().item()

    def __call__(self,preds : Dict[str,torch.Tensor], targets : Dict[str,torch.Tensor]) -> Tuple[List[float]]:
        '''
        Compute phonological and semantic accuracies for all
        values of [k]/[tau]
        
        Args:
            preds (Dict[str,torch.Tensor]) : predicted phonology and semantics
            targets (Dict[str,torch.Tensor]) : ground-truth phonology and semantics
        Returns:
            Accuracies
        '''
        phonology_preds = preds['phonology']
        phonology_targets = targets['phonology']

        semantics_preds = preds['semantics']
        semantics_targets = targets['semantics']

        phon_accuracy = [[self.compute_phon_accuracy(phonology_pred,phonology_targets,k) 
                            for k in self.k] for phonology_pred in phonology_preds]
        sem_accuracy = [[self.compute_sem_accuracy(semantics_pred,semantics_targets,tau)
                            for tau  in self.tau] for semantics_pred in semantics_preds]
        return phon_accuracy,sem_accuracy

class TrainerConfig:
    def __init__(self,**kwargs):
        '''
        Configuration class to store training options / hyperparameters. 
        Somewhat limited ATM.
        
        Args:
            **kwargs (dict)
        '''
        self.params = kwargs
        
    @classmethod
    def from_json(cls,json_path : str) -> 'TrainerConfig':
        '''
        Read config parameters from .json file
        
        Args:
             json_path (str) : path to config file
          Return:
             TrainerConfig
        '''
        config_params = json.load(open(json_path,'r'))
        return cls(**config_params)
        
    def create_trainer(self,phoneme_embedding_matrix : torch.Tensor) -> 'Trainer':
        '''
        Instantiate Trainer w/ desired parameters
        
        Args:
            phoneme_embedding_matrix (torch.Tensor) : feature vectors corresponding
                                                      to each phoneme
        Returns:
            Trainer
        '''
        
        ### TODO: Add support for additional sovlers besides Forward Euler
        if self.params.get('solver','forward_euler') == 'forward_euler':
            solver = forward_euler
        else:
            raise ValueError('Supported solvers include: forward_euler')
        
        ### Set Zero Error Radius
        zer = self.params.get('zer',.1)

        ### Set initial conditions
        init_val = self.params.get('init_val',-15)

        return Trainer(solver,phoneme_embedding_matrix,zer,init_val)
    
class Trainer:
    def __init__(self,solver : Callable[[torch.nn.Module,Dict[str,torch.Tensor],float,float,float],
                            Dict[str,torch.Tensor]], phoneme_embedding_matrix : torch.Tensor, 
                            zer : float, init_val : float):
        '''
        Utility class for running/updating the Triangle Model 
        under various lesioning configurations.
         
        Args:
            solver (Callable[[torch.nn.Module,Dict[str,torch.Tensor],float,float,float],
                            Dict[str,torch.Tensor]]) : ODE solver routine 
            phoneme_embedding_matrix (torch.Tensor) : feature vectors corresponding
                                                      to each phoneme
            zer (float) : Zero Error Radiues for Cross-Entropy loss.
            bound (float) : bound on the inverse sigmoid
            init_val (float) : all unspecified states / hidden units will be initialized
                               to this value.
        '''

        self.solver = solver
        self.metrics = Metrics(phoneme_embedding_matrix,[1,2,3],[.4,.5,.6])
        self.zer = zer
        self.init_val = init_val

        self.device = torch.device('cpu')
        
    def to(self,device : torch.device) -> 'Trainer':
        '''
        Move stored tensors to the provided device.
        
        Args:
            device (torch.device)
        Returns:
            self
        '''
        assert isinstance(device,torch.device)
        self.device = device
        self.metrics.to(device)
        return self

    def cross_entropy(self,preds : torch.Tensor, targets : torch.Tensor,
                      zer : float, eps : Optional[float] = 1e-4) -> torch.Tensor:
        '''
        Compute Cross-Entropy loss w/ ZER
        
        Args:
            preds (torch.Tensor) : predicted vectors
            targets (torch.Tensor) : ground-truth vectors
            zer (float) : Zero Error Radius
            eps (Optional[float]) : Added for numerical stability
        Returns:
            Value of loss
        '''
        mask = ((targets-preds).abs()>=zer).float()

        cross_entropy = -targets * (eps + preds).log()
        cross_entropy = cross_entropy - (1-targets) * (1 + eps - preds).log()
        return (mask * cross_entropy).sum(dim=(-1,-2))/(eps + mask.sum(dim=(-1,-2)))

    def collate_outputs(self,outputs : Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
       '''
       Compute the "output" of the Triangle Model.
       
       Args:
           outputs (Dict[str,torch.Tensor]) : Values of all states / hidden units
                                              at each timestep.
       Returns:
           Phonology and semantics (all timesteps) normalized via a logistic sigmoid. 
       '''
       for idx,output in enumerate(outputs):
           S = clipped_sigmoid(output['semantics'],BOUND)[None]
           P = clipped_sigmoid(output['phonology'],BOUND)[None]

           if idx == 0:
              semantics = S
              phonology = P
           else:
              semantics = torch.cat((semantics,S),dim=0)
              phonology = torch.cat((phonology,P),dim=0)

       return phonology,semantics
 
    def create_inputs(self,model : TriangleModel, data : Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
       '''
       Define initial conditions
       
       Args:
           model (TriangleModel) : Only used to retrieve dimensionality of
                                   states / hidden units
           data (Dict[str,torch.Tensor]) : For all states / hidden units in [data],
                                           we set the corresponding intial conditions 
                                           to the dictionary values.
       Return:
           Initial conditions for all states / hidden units 
       '''
    
       ### Determine batch size
       temp = torch.zeros((0,))
       batch_size = max([
                        data.get('orthography',temp).shape[0],
                        data.get('phonology',temp).shape[0],
                        data.get('semantics',temp).shape[0],
                    ])

       ### Define orthography (constant)
       if data.get('orthography',None) is not None:
          inputs = {'orthography':invert_binary(data['orthography'])}
       else:
          inputs = {'orthography': self.init_val * torch.ones((batch_size,model.orth_dim),device=self.device)}

       ### Define initial phonology
       if data.get('phonology',None) is not None:
          inputs['phonology'] = invert_binary(data['phonology'])
       else:
          inputs['phonology'] = self.init_val * torch.ones((batch_size,model.phon_dim),device=self.device)

       ### Define initial semantics
       if data.get('semantics',None) is not None:
          inputs['semantics'] = invert_binary(data['semantics'])
       else:
          inputs['semantics'] = self.init_val * torch.ones((batch_size,model.sem_dim),device=self.device)

       ### Define initial conditions for cleanup units
       inputs['cleanup_phon'] = self.init_val * torch.ones((batch_size,model.phon_cleanup_dim),device=self.device)
       inputs['cleanup_sem'] = self.init_val * torch.ones((batch_size,model.sem_cleanup_dim),device=self.device)

       ### Define initial conditions for oral units
       inputs['sem_2_phon'] = self.init_val * torch.ones((batch_size,model.sem_2_phon_dim),device=self.device)
       inputs['phon_2_sem'] = self.init_val * torch.ones((batch_size,model.phon_2_sem_dim),device=self.device)

       ### Define initial conditions for reading units
       inputs['orth_2_phon'] = self.init_val * torch.ones((batch_size,model.orth_2_phon_dim),device=self.device)
       inputs['orth_2_sem'] = self.init_val * torch.ones((batch_size,model.orth_2_sem_dim),device=self.device)

       return inputs

    def step(self,model : TriangleModel, inputs : Dict[str,torch.Tensor],
             targets : Optional[Dict[str,torch.Tensor]] = None,
             opt : Optional[torch.optim.Optimizer] = None,
             **kwargs) -> Union[
                                Tuple[torch.Tensor,torch.Tensor],
                                Tuple[List[float],List[float]],
                                Tuple[torch.Tensor,torch.Tensor,List[float],List[float]],
                          ]:
        '''
        Perform a single forward pass through the Triangle Model w/
        the specified training parameters. 
        
        Args:
            model (TriangleModel)
            inputs (Dict[str,torch.Tensor]) : specified initial conditions
            targets (Optional[Dict[str,torch.Tensor]]) : ground-truth phonology and 
                                                         semantics.
            opt (Optional[torch.optim.Optimizer]) : optimizer for gradient descent
            
        Returns:
            If [targets] are not given, returns the output of [model]
            If [targets] are given:
                - If [opt] is given:
                    Return losses and accuracies
                - If [opt] is not given:
                    Return accuracies
                    
        '''
        
        ### Get timestep at which to begin computing the error
        start_error = kwargs.get('start_error',2)
        
        ### Get solver paramters
        delta_t = kwargs.get('delta_t',1/3)
        t_0 = kwargs.get('t_0',0)
        T = kwargs.get('T',4)

        detach = kwargs.get('detach',False)

        ### Define initial conditions
        inputs = self.create_inputs(model,inputs)
        
        ### Call solver
        outputs = self.solver(model,inputs,t_0,T,delta_t,detach)
        
        ### Extract outputs
        predicted_phonology,predicted_semantics = self.collate_outputs(outputs)

        ### If no targets are provided, we just return the output 
        ### of the model
        if targets is None:
           if kwargs.get('return_outputs',False):
              return outputs
           else:
              return predicted_phonology,predicted_semantics

        else: 
            phonology = targets['phonology']
            semantics = targets['semantics']
 
            ### Compute accuracies for the final timestep
            p_acc,s_acc = self.metrics(
                          {'phonology':predicted_phonology[start_error::],
                           'semantics':predicted_semantics[start_error::]},
                          targets)

            if opt is None:
               return None,None,p_acc,s_acc

            ### If an optimizer is provided, perform a training step.
            else:
                
               ### Compute losses
               phonology_loss = self.cross_entropy(predicted_phonology[start_error::],phonology[None],self.zer)
               semantics_loss = self.cross_entropy(predicted_semantics[start_error::],semantics[None],self.zer)

               ### Set timestep weighting.
               ### TODO: allow user to change weighting; must be monotonic
               weighting = torch.arange(1,phonology_loss.shape[0]+1,device=self.device)
               weighting = torch.clamp(weighting/weighting[-1],.5,1)

               summed_phonology_loss = (weighting * phonology_loss).sum()
               summed_semantics_loss = (weighting * semantics_loss).sum()

               ### BPTT
               loss = summed_phonology_loss + summed_semantics_loss
               loss.backward()

               ### Update parameters
               opt.step()
               opt.zero_grad()

               return phonology_loss.tolist(),semantics_loss.tolist(),p_acc,s_acc
        
    def train_p2p(self,model : TriangleModel, opt : torch.optim.Optimizer,
                data : Dict[str,torch.Tensor]) -> Tuple[float,List[float]]:
        '''
        Run + update the phonological cleanup units.
        
        Args:
            model (TriangleModel)
            data (Dict[str,torch.Tensor]) : phonological and semantic representations
                                            for a set of words
            opt (torch.optim.Optimizer) : optimizer for gradient descent
        '''
        model.set_lesions(['o2s','o2p','p2s','s2p','s2s'])

        start_error = -4
        t_0 = 2 + 2/3

        inputs = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }

        targets = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }

        phon_loss,sem_loss,phon_acc,sem_acc = self.step(model,inputs,opt=opt,targets=targets,
                                                          start_error = start_error, t_0 = t_0)
        model.set_lesions()
        return phon_loss,phon_acc

    def train_s2s(self,model : TriangleModel, opt : torch.optim.Optimizer,
                data : Dict[str,torch.Tensor]) -> Tuple[float,List[float]]:
        '''
        Run + update the semantic cleanup units
        
        Args:
            model (TriangleModel)
            data (Dict[str,torch.Tensor]) : phonological and semantic representations
                                            for a set of words
            opt (torch.optim.Optimizer) : optimizer for gradient descent
        '''
        model.set_lesions(['o2s','o2p','p2s','s2p','p2p'])

        start_error = -4
        t_0 = 2 + 2/3

        inputs = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }

        targets = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }

        phon_loss,sem_loss,phon_acc,sem_acc =  self.step(model,inputs,opt=opt,targets=targets,
                                                          start_error = start_error, t_0 = t_0)
        model.set_lesions()
        return sem_loss,sem_acc

    def train_s2p(self,model : TriangleModel, opt : torch.optim.Optimizer,
                data : Dict[str,torch.Tensor]) -> Tuple[float,List[float]]:
        '''
        Run + update the S2P oral pathway and phonological cleanup units
        
        Args:
            model (TriangleModel)
            data (Dict[str,torch.Tensor]) : phonological and semantic representations
                                            for a set of words
            opt (torch.optim.Optimizer) : optimizer for gradient descent
        '''
        
        model.set_lesions(['o2s','o2p','p2s','s2s'])

        start_error = -3
        t_0 = 0

        inputs = {'semantics':data['semantics'].to(self.device)}
        targets = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }
        loss,_,acc,_ =  self.step(model,inputs,opt=opt,targets=targets,
                                   start_error = start_error, t_0 = t_0)
        model.set_lesions()
        return loss,acc

    def train_p2s(self,model : TriangleModel, opt : torch.optim.Optimizer,
                data : Dict[str,torch.Tensor]) -> Tuple[float,List[float]]:
        '''
        Run + update the P2S oral pathway and semantic cleanup units
        
        Args:
            model (TriangleModel)
            data (Dict[str,torch.Tensor]) : phonological and semantic representations
                                            for a set of words
            opt (torch.optim.Optimizer) : optimizer for gradient descent
        '''
        model.set_lesions(['o2s','o2p','s2p','p2p'])

        start_error = -3
        t_0 = 0

        inputs = {'phonology':data['phonology'].to(self.device)}
        targets = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }
        _,loss,_,acc = self.step(model,inputs,opt=opt,targets=targets,
                                   start_error = start_error, t_0 = t_0)
        model.set_lesions()
        return loss,acc

    def train_full(self,model : TriangleModel, opt : torch.optim.Optimizer,
                data : Dict[str,torch.Tensor], lesions = [],
                detach : Optional[bool] = False) -> Tuple[Tuple[float],Tuple[List[float]]]:
        '''
        Run + update the "full" (i.e: w/ orthography, no lesions 
        by default) Triangle Model.
        
        Args:
            model (TriangleModel)
            data (Dict[str,torch.Tensor]) : phonological and semantic representations
                                            for a set of words
            opt (torch.optim.Optimizer) : optimizer for gradient descent
        '''
        
        model.set_lesions(lesions)
        start_error = 1
        t_0 = 0

        inputs = {'orthography':data['orthography'].to(self.device)}
        targets = {
                    'phonology':data['phonology'].to(self.device),
                    'semantics':data['semantics'].to(self.device),
                  }
        phon_loss,sem_loss,phon_acc,sem_acc = self.step(model,inputs,opt=opt,targets=targets,
                                                          start_error = start_error, 
                                                          t_0 = t_0,detach=detach)

        model.set_lesions()
        return (phon_loss,sem_loss),(phon_acc,sem_acc)
