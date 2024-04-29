import torch
import json
import os
from typing import Optional,List,Dict
from dataclasses import dataclass

BOUND = int(os.environ.get('BOUND','15'))

def clipped_sigmoid(X,bound):
    nX = torch.sigmoid(X)
    nX = nX - (X<=-bound).float() * nX \
            + (X>=bound).float() * (1 - nX)
    return nX

class TimeAveragedInputs(torch.nn.Module):
   def __init__(self,in_features_list : List[int], out_features : int,bias : Optional[bool] = False):
       '''
        Time-Averaged Inputs formulation of the gradient (refer to
        Plaut et. al, 1998).

        Args:
           in_features_list (List[int]) : dimemsionality of each input vector
           out_features (int) : dimensionality of output vectors
           bias (bool) : if True, use trainable bias term

       '''
       super(TimeAveragedInputs,self).__init__()
       self.weights = torch.nn.ModuleList(
                          [torch.nn.Linear(in_features,out_features,bias=bias) for in_features in in_features_list]   
                          )
       for weight in self.weights:
           torch.nn.init.uniform_(weight.weight,-.1,.1)

   def forward(self, X : List[torch.Tensor], Y : torch.Tensor, lesions : List[int]) -> torch.Tensor:
       '''
         Compute the gradient of [Y] w.r.t to time. 
         
         Args:
            X (List[torch.Tensor]) : dimensionalities [in_features_list]
            Y (torch.Tensor) : dimensionality [out_features]
            lesions (List[int]) : binary lesions which are applied to each 
                                  Tensor in [X].
            
         Returns:
            gradient; dimensionality [out_features]
       '''
       assert len(lesions) == len(X)
       if sum(lesions) == len(lesions):
          return 0
      
       nX = 0
       for idx,input_vector in enumerate(X):
           if lesions[idx] != 1:
               nX = nX + self.weights[idx](clipped_sigmoid(input_vector,BOUND))
       return nX - Y


@dataclass()
class ModelConfig:
    '''
       Configuration class to define TriangleModel architecture.
       
       Args:
           learn_bias (bool) : whether to include bias term in gradient computation.
           
           orth_dim (int) : dimensionality of the orthographic state
           phon_dim (int) : dimensionality of the phonological state
           sem_dim (int) : dimensionality of the semantic state
           
           phon_cleanup_dim (int) : dimensionality of the phonological cleanup unit
           sem_cleanup_dim (int) : dimensionality of the semantic cleanup unit
           
           phon_2_sem_dim (int) : dimensionality of the phonology -> semantics 
                                  hidden unit
           sem_2_phon_dim (int) : dimensionality of the semantics -> phonology
                                  hidden unit
                                 
           orth_2_sem_dim (int) : dimensionality of the orthography -> semantics
                                  hidden unit.
           orth_2_phon_dim (int) : dimensionality of the orthography -> phonology
                                   hidden unit.
    '''
    
    learn_bias : bool = False

    orth_dim : int = 111
    phon_dim : int = 200
    sem_dim : int = 1989

    phon_cleanup_dim : int = 50
    sem_cleanup_dim : int = 50

    phon_2_sem_dim : int = 500
    sem_2_phon_dim : int = 500

    orth_2_sem_dim : int = 500
    orth_2_phon_dim : int = 100

    @classmethod
    def from_json(cls,json_path : str) -> 'ModelConfig':
        '''
          Read config parameters from .json file
          
          Args:
             json_path (str) : path to config file
          Return:
             ModelConfig
        '''
        config_params = json.load(open(json_path,'r'))
        return cls(**config_params)
        
    def create_model(self,operator : Optional[torch.nn.Module] = TimeAveragedInputs,
                          lesions : Optional[List[str]] = []) -> 'TriangleModel':
        '''
          Instantiate TriangleModel w/ desired parameters
          
          Args:
              operator (torch.nn.Module): module to compute gradient contributions.
                                          Defaults to TimeAveragedInputs.
              lesions (List[str]): list of lesions to apply to the model. Accepts
                                   values of 'p2p', 's2s', 'p2s', 's2p', 'o2s', 
                                   and 'o2p'.
              
          Return:
              TriangleModel 
        '''
        if self.learn_bias == "False":
           bias = False
        else: bias = True
        print(self.learn_bias,bias)
        return TriangleModel(self.orth_dim,self.phon_dim,self.sem_dim,
                                self.phon_cleanup_dim,self.sem_cleanup_dim,
                                self.phon_2_sem_dim,self.sem_2_phon_dim,
                                self.orth_2_sem_dim,self.orth_2_phon_dim,
                                bias,operator,lesions)

class TriangleModel(torch.nn.Module):
    def __init__(self, orth_dim : int, phon_dim : int, sem_dim : int,
                    phon_cleanup_dim : int, sem_cleanup_dim : int,
                    phon_2_sem_dim : int, sem_2_phon_dim : int,
                    orth_2_sem_dim : int, orth_2_phon_dim : int,
                    learn_bias : bool, operator : torch.nn.Module,
                    lesions : List[str]):
        super(TriangleModel,self).__init__()
        '''
          A PyTorch implemtation of the Triangle Model detailed in
          Harm and Seidenberg, 2004.
          
          Args:           
              orth_dim (int) : dimensionality of the orthographic state
              phon_dim (int) : dimensionality of the phonological state
              sem_dim (int) : dimensionality of the semantic state

              phon_cleanup_dim (int) : dimensionality of the phonological cleanup unit
              sem_cleanup_dim (int) : dimensionality of the semantic cleanup unit

              phon_2_sem_dim (int) : dimensionality of the phonology -> semantics 
                                     hidden unit
              sem_2_phon_dim (int) : dimensionality of the semantics -> phonology
                                     hidden unit

              orth_2_sem_dim (int) : dimensionality of the orthography -> semantics
                                     hidden unit.
              orth_2_phon_dim (int) : dimensionality of the orthography -> phonology
                                      hidden unit.
                                      
              learn_bias (bool) : whether to include bias term in gradient computation.
              
              operator (torch.nn.Module): module to compute gradient contributions.
                                          Defaults to TimeAveragedInputs.
                                          
              lesions (List[str]): list of lesions to apply to the model. Accepts
                                   values of 'p2p', 's2s', 'p2s', 's2p', 'o2s', 
                                   and 'o2p'.

        '''
        self.set_lesions(lesions)

        self.orth_dim,self.phon_dim,self.sem_dim = orth_dim,phon_dim,sem_dim
        self.phon_cleanup_dim,self.sem_cleanup_dim = phon_cleanup_dim,sem_cleanup_dim
        self.phon_2_sem_dim,self.sem_2_phon_dim = phon_2_sem_dim,sem_2_phon_dim
        self.orth_2_sem_dim,self.orth_2_phon_dim = orth_2_sem_dim,orth_2_phon_dim

        ### Instantiate phonology gradient
        self.phon_gradient = operator([phon_cleanup_dim,sem_2_phon_dim,
                                    orth_2_phon_dim,orth_dim],phon_dim,learn_bias)

        ### Instantiate semantics gradient
        self.sem_gradient = operator([sem_cleanup_dim,phon_2_sem_dim,
                                    orth_2_sem_dim,orth_dim],sem_dim,learn_bias)

        ### Instantiate cleanup gradients
        self.p2p_gradient = operator([phon_dim],phon_cleanup_dim,learn_bias)
        self.s2s_gradient = operator([sem_dim],sem_cleanup_dim,learn_bias)

        ### Instantiate oral hidden unit gradients
        self.s2p_gradient = operator([sem_dim],sem_2_phon_dim,learn_bias)
        self.p2s_gradient = operator([phon_dim],phon_2_sem_dim,learn_bias)

        ### Instantiate reading hidden unit gradients
        self.o2p_gradient = operator([orth_dim],orth_2_phon_dim,learn_bias)
        self.o2s_gradient = operator([orth_dim],orth_2_sem_dim,learn_bias)

    def set_lesions(self,lesions = []):
        if 'o2p' in lesions:
           self.o2p_lesion = 1
        else:
           self.o2p_lesion = 0

        if 'o2s' in lesions:
           self.o2s_lesion = 1
        else:
           self.o2s_lesion = 0

        if 'p2s' in lesions:
           self.p2s_lesion = 1
        else:
           self.p2s_lesion = 0

        if 's2p' in lesions:
           self.s2p_lesion = 1
        else:
           self.s2p_lesion = 0

        if 's2s' in lesions:
           self.s2s_lesion = 1
        else:
           self.s2s_lesion = 0

        if 'p2p' in lesions:
           self.p2p_lesion = 1
        else:
           self.p2p_lesion = 0

    def forward(self,inputs : Dict[str,torch.Tensor],**kwargs) -> Dict[str,torch.Tensor]:
        '''
           Compute gradients of all states / hidden units w.r.t to time.
           
           Args:
              inputs (Dict[str,torch.Tensor]) : Values of all states / hidden units
                                                at the current timestep.
           Returns:
              Gradients of all states / hidden units
        '''
        
        detach = kwargs.get('detach',False)

        ### Get states
        orthography = inputs['orthography']
        phonology = inputs['phonology']
        semantics = inputs['semantics'] 

        ### Get cleanup units
        cleanup_phon = inputs['cleanup_phon']
        cleanup_sem = inputs['cleanup_sem']

        if detach:
           cleanup_phon = cleanup_phon.detach()
           cleanup_sem = cleanup_sem.detach()

        ### Get oral hidden units
        phon_2_sem = inputs['phon_2_sem']
        sem_2_phon = inputs['sem_2_phon']

        if detach:
           sem_2_phon = sem_2_phon.detach()
           phon_2_sem = phon_2_sem.detach()

        ### Get reading hidden units
        orth_2_sem = inputs['orth_2_sem']
        orth_2_phon = inputs['orth_2_phon']

        ### Get lesions
        p2p_lesion = self.p2p_lesion
        s2s_lesion = self.s2s_lesion
        
        s2p_lesion = self.s2p_lesion
        p2s_lesion = self.p2s_lesion
        
        o2p_lesion = self.o2p_lesion
        o2s_lesion = self.o2s_lesion

        ### Compute gradient of phonology
        phon_gradient = self.phon_gradient([cleanup_phon,sem_2_phon,orth_2_phon,orthography],phonology,
                                                          [p2p_lesion,s2p_lesion,o2p_lesion,o2p_lesion])
        ### Compute gradient of semantics
        sem_gradient = self.sem_gradient([cleanup_sem,phon_2_sem,orth_2_sem,orthography],semantics,
                                                     [s2s_lesion,p2s_lesion,o2s_lesion,o2s_lesion])

        ### Compute gradient of cleanup units
        cleanup_phon_gradient = self.p2p_gradient([phonology],cleanup_phon,[p2p_lesion])
        cleanup_sem_gradient = self.s2s_gradient([semantics],cleanup_sem,[s2s_lesion])

        ### Compute gradient of oral hidden units
        phon_2_sem_gradient = self.p2s_gradient([phonology],phon_2_sem,[p2s_lesion])
        sem_2_phon_gradient = self.s2p_gradient([semantics],sem_2_phon,[s2p_lesion])

        ### Compute gradient of reading hidden units
        orth_2_sem_gradient = self.o2s_gradient([orthography],orth_2_sem,[o2s_lesion])
        orth_2_phon_gradient = self.o2p_gradient([orthography],orth_2_phon,[o2p_lesion])

        ### Write gradients to dictionary
        gradients = {}

        gradients['orthography'] = torch.zeros_like(orthography,device=orthography.device)
        gradients['phonology'] = phon_gradient
        gradients['semantics'] = sem_gradient

        gradients['cleanup_phon'] = cleanup_phon_gradient
        gradients['cleanup_sem'] = cleanup_sem_gradient

        gradients['phon_2_sem'] = phon_2_sem_gradient
        gradients['sem_2_phon'] = sem_2_phon_gradient

        gradients['orth_2_sem'] = orth_2_sem_gradient
        gradients['orth_2_phon'] = orth_2_phon_gradient

        return gradients
