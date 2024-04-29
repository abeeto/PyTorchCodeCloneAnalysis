import torch
import os
import glob
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

from train import forward_euler,TrainerConfig,Trainer
from dataset import Monosyllabic_Dataset
from model import ModelConfig,TriangleModel

from typing import List,Optional,Dict,Union

import pandas as pd
import tqdm.auto

def plot_performance(ID : str):
    '''
    Plot and save accuracies.
    
    Args:
        ID (str) : model ID. 
    '''
    os.makedirs(f'figures/{ID}',exist_ok=True)
    
    p2s = np.array(np.load(f'metrics/{ID}/eval_p2s_acc.npy'))
    s2p = np.array(np.load(f'metrics/{ID}/eval_s2p_acc.npy')) 

    fig = plt.figure(figsize=(15,8),dpi=400)
    plt.subplot(1,2,1)

    plt.plot(s2p[:,0])
    plt.plot(s2p[:,1])
    plt.plot(s2p[:,2])

    plt.legend(title='k',labels=['1','2','3'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Semantics -> Phonology')

    plt.subplot(1,2,2)

    plt.plot(p2s[:,0])
    plt.plot(p2s[:,1])
    plt.plot(p2s[:,2])

    plt.legend(title='Threshold',labels=['.4','.5','.6'])
    plt.xlabel('Iteration')
    plt.title('Phonology -> Semantics');
    
    plt.savefig(f'figures/{ID}/phonology_semantics.png')
    
    o2s = np.array(np.load(f'metrics/{ID}/eval_o2s_acc.npy'))
    o2p = np.array(np.load(f'metrics/{ID}/eval_o2p_acc.npy')) 

    plt.figure(figsize=(15,8),dpi=400)
    plt.subplot(1,2,1)

    plt.plot(o2p[:,0])
    plt.plot(o2p[:,1])
    plt.plot(o2p[:,2])

    plt.legend(title='k',labels=['1','2','3'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Orthography -> Phonology')

    plt.subplot(1,2,2)

    plt.plot(o2s[:,0])
    plt.plot(o2s[:,1])
    plt.plot(o2s[:,2])

    plt.legend(title='Threshold',labels=['.4','.5','.6'])
    plt.xlabel('Iteration')
    plt.title('Orthography -> Semantics');
    
    plt.savefig(f'figures/{ID}/orthography.png')

    plt.figure(figsize=(15,8))
    plt.plot(o2s[:,1],label='Full')

    o2p2s = np.array(np.load(f'metrics/{ID}/eval_o2p2s_acc.npy'))
    o2s_only = np.array(np.load(f'metrics/{ID}/eval_o2s_only_acc.npy')) 

    plt.plot(o2p2s[:,1],label='O->P->S')
    plt.plot(o2s_only[:,1],label='O->S')

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(title='Variants')

    plt.savefig(f'figures/{ID}/semantic_evolution.png')
    plt.close('all')

def plot_taraban(results : np.array ,filename : str):
    '''
    Plot and save results on Taraban and McClelland, 1987.
    
    Args:
        results (np.array) : array containing data statistics and the 
                             model's SSE on the Taraban corpus. 
        filename (str) : base name at which to save figures. 
    '''
    fig = plt.figure(figsize=(15,8),dpi=400)
    plt.plot([0,1],[np.mean(results[:,2][np.logical_and(results[:,1]==1,results[:,0]==0)]),
                    np.mean(results[:,2][np.logical_and(results[:,1]==1,results[:,0]==1)])],
             label='Regular',color='blue')
    plt.plot([0,1],[np.mean(results[:,2][np.logical_and(results[:,1]==0,results[:,0]==0)]),
                    np.mean(results[:,2][np.logical_and(results[:,1]==0,results[:,0]==1)])],
             label='Exception',color='orange')
    plt.ylabel('SSE')
    plt.legend()
    plt.title('Taraban Mean Error')
    plt.xticks([0,1],['Low Frq','High Frq']);
    
    plt.savefig(f'{filename}_mean.png')
    
    plt.figure(figsize=(15,8),dpi=400)
    plt.plot([0,1],[np.median(results[:,2][np.logical_and(results[:,1]==1,results[:,0]==0)]),
                    np.median(results[:,2][np.logical_and(results[:,1]==1,results[:,0]==1)])],
             label='Regular',color='blue')
    plt.plot([0,1],[np.median(results[:,2][np.logical_and(results[:,1]==0,results[:,0]==0)]),
                    np.median(results[:,2][np.logical_and(results[:,1]==0,results[:,0]==1)])],
             label='Exception',color='orange')
    plt.ylabel('SSE')
    plt.legend()
    plt.title('Taraban Median Error')
    plt.xticks([0,1],['Low Frq','High Frq']);
    
    plt.savefig(f'{filename}_median.png')
    plt.close('all')
    
def eval_taraban(ckpts : List[str], model : TriangleModel,trainer : Trainer):
    '''
    Compute results on Taraban and McClelland (1987).
    
    Args:
        ckpts (List[str]) : list of model checkpoints to evaluate
        model (TriangleModel)
        trainer (Trainer)
    '''
    
    taraban = pd.read_csv('datasets/JasonLo/taraban.csv')

    orthography,phonology = taraban['ort'],taraban['pho']
    cond = taraban['cond']
    frq,con = [],[]
    for c in cond:
        frq.append(float('High' in c))
        con.append(float('regular' in c))
        
    global orthography_tokenizer,phonology_tokenizer
        
    device = trainer.device
    ### Iterate over checkpoints
    for ckpt in tqdm.tqdm(ckpts[-1::],position=0):
       model.load_state_dict(torch.load(ckpt,map_location=device))
       model.eval()
        
       ID = ckpt.split('/')[1]
       step = ckpt.split('/')[-1].replace('.pth','')
       os.makedirs(f'figures/{ID}/taraban/{step}',exist_ok=True)
       with torch.no_grad():
           ### Iterate over starting timestep for SSE
           for start_step in range(2,12):
               results = []
               for idx in tqdm.tqdm(range(len(orthography)),position=0):
                   orthography_tensor = orthography_tokenizer(orthography.iloc[idx])[None].to(device)
                   phonology_tensor = phonology_tokenizer(phonology.iloc[idx])[None].to(device)
                   
                   predicted_phonology,_ = trainer.step(model,{'orthography':orthography_tensor})
                   sse = (phonology_tensor-predicted_phonology[start_step::]).pow(2).sum(dim=(1,2)).mean().item()
                   results.append([frq[idx],con[idx],sse])
                    
               results = np.array(results)
               plot_taraban(results,f'figures/{ID}/taraban/{step}/{start_step}')
             
                
def plot_strain(results : np.array, filename : str):
    '''
    Plot and save results on Strain, et. al (1995).
    
    Args:
        results (np.array) : array containing data statistics and the 
                             model's SSE on the Strain corpus. 
        filename (str) : base name at which to save figures. 
    '''
    fig = plt.figure(figsize=(15,8),dpi=400)
    plt.plot([0,1],[np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==0),results[:,0]==0)]),
                    np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==0),results[:,0]==1)])],
             label='INC-LI',color='blue')
    plt.plot([0,1],[np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==0),results[:,0]==0)]),
                    np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==0),results[:,0]==1)])],
             label='INC-HI',color='blue',linestyle=':')
    plt.plot([0,1],[np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==1),results[:,0]==0)]),
                    np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==1),results[:,0]==1)])],
             label='CON-LI',color='orange')
    plt.plot([0,1],[np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==1),results[:,0]==0)]),
                    np.mean(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==1),results[:,0]==1)])],
             label='CON-HI',color='orange',linestyle=':')
    plt.ylabel('SSE')
    plt.legend()
    plt.title('Strain Mean Error')
    plt.xticks([0,1],['Low Frq','High Frq']);
    
    plt.savefig(f'{filename}_mean.png')
    
    plt.figure(figsize=(15,8),dpi=400)
    plt.plot([0,1],[np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==0),results[:,0]==0)]),
                    np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==0),results[:,0]==1)])],
             label='INC-LI',color='blue')
    plt.plot([0,1],[np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==0),results[:,0]==0)]),
                    np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==0),results[:,0]==1)])],
             label='INC-HI',color='blue',linestyle=':')
    plt.plot([0,1],[np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==1),results[:,0]==0)]),
                    np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==0,results[:,1]==1),results[:,0]==1)])],
             label='CON-LI',color='orange')
    plt.plot([0,1],[np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==1),results[:,0]==0)]),
                    np.median(results[:,3][np.logical_and(np.logical_and(results[:,2]==1,results[:,1]==1),results[:,0]==1)])],
             label='CON-HI',color='orange',linestyle=':')
    plt.ylabel('SSE')
    plt.legend()
    plt.title('Strain Median Error')
    plt.xticks([0,1],['Low Frq','High Frq']);
    
    plt.savefig(f'{filename}_median.png')
    plt.close('all')
    
def eval_strain(ckpts : List[str], model : TriangleModel, trainer : Trainer):
    '''
    Compute results on Strain (1995).
    
    Args:
        ckpts (List[str]) : list of model checkpoints to evaluate
        model (TriangleModel)
        trainer (Trainer)
    '''
    
    strain = pd.read_csv('datasets/JasonLo/strain.csv')

    orthography,phonology = strain['ort'],strain['pho']
    frq,con = strain['frequency'],strain['pho_consistency']
    frq,con = (frq=='HF').astype(float),(con=='CON').astype(float)
    img = strain['imageability']
    img = (img=='HI').astype(float)

    global orthography_tokenizer,phonology_tokenizer
    
    device = trainer.device

    ### Iterate over checkpoints
    for ckpt in tqdm.tqdm(ckpts[-1::],position=0):
       model.load_state_dict(torch.load(ckpt,map_location=device))
       model.eval()
        
       ID = ckpt.split('/')[1]
       step = ckpt.split('/')[-1].replace('.pth','')
       os.makedirs(f'figures/{ID}/strain/{step}',exist_ok=True)
       with torch.no_grad():
           ### Iterate over starting timestep for SSE
           for start_step in range(2,12):
               results = []
               for idx in tqdm.tqdm(range(len(orthography)),position=0):
                   orthography_tensor = orthography_tokenizer(orthography.iloc[idx])[None].to(device)
                   phonology_tensor = phonology_tokenizer(phonology.iloc[idx])[None].to(device)
                   
                   predicted_phonology,_ = trainer.step(model,{'orthography':orthography_tensor})
                   sse = (phonology_tensor-predicted_phonology[start_step::]).pow(2).sum(dim=(1,2)).mean().item()
                   results.append([frq[idx],con[idx],img[idx],sse])
                    
               results = np.array(results)
               plot_strain(results,f'figures/{ID}/strain/{step}/{start_step}')

    
def plot_glushko(results : np.array, filename : str):
    '''
    Plot and save results on Glushko (1979).
    
    Args:
        results (np.array) : array containing data statistics and the 
                             model's SSE on the Glushko nonwords. 
        filename (str) : base name at which to save figures. 
    '''
    fig = plt.figure(figsize=(15,8),dpi=400)
    plt.plot([0,1],[np.mean(results[:,1][results[:,0]==0]),
                    np.mean(results[:,1][results[:,0]==1])],
             label='Short-Grain',color='blue')
    plt.plot([0,1],[np.mean(results[:,2][results[:,0]==0]),
                    np.mean(results[:,2][results[:,0]==1])],
             label='Long-Grain',color='orange')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Glushko Nonwords')
    plt.xticks([0,1],['Ambiguous','Unambiguous']);
    
    plt.savefig(f'{filename}.png')
    plt.close('all')    

def eval_glushko(ckpts : List[str], model : TriangleModel, trainer : Trainer):
    '''
    Compute results on Glushko nonwords (1979).
    
    Args:
        ckpts (List[str]) : list of model checkpoints to evaluate
        model (TriangleModel)
        trainer (Trainer)
    '''
    
    nonwords = pd.read_csv('datasets/JasonLo/nonwords.csv')[:86]
    orthography = nonwords['ort']
    short_grain = nonwords['pho_small']
    long_grain = nonwords['pho_large']
    
    cond = nonwords['condition']
    cond = (cond=='unambiguous').astype(float)
    
    device = trainer.device

    global orthography_tokenizer,phonology_tokenizer
    for ckpt in tqdm.tqdm(ckpts[-1::],position=0):
       model.load_state_dict(torch.load(ckpt))
       model.eval()
        
       ID = ckpt.split('/')[1]
       step = ckpt.split('/')[-1].replace('.pth','')
       os.makedirs(f'figures/{ID}/glushko',exist_ok=True)
       with torch.no_grad():
              results = []
              for idx in tqdm.tqdm(range(len(orthography)),position=0):
                   orthography_tensor = orthography_tokenizer(orthography.iloc[idx])[None].to(device)
                   short_phonology_tensor = phonology_tokenizer(short_grain.iloc[idx])[None].to(device)
                   long_phonology_tensor = phonology_tokenizer(long_grain.iloc[idx])[None].to(device)

                   predicted_phonology,_ = trainer.step(model,{'orthography':orthography_tensor})
                    
                   short_acc = trainer.metrics.compute_phon_accuracy(predicted_phonology,short_phonology_tensor,1)
                   long_acc = trainer.metrics.compute_phon_accuracy(predicted_phonology,long_phonology_tensor,1)
                    
                   results.append([cond[idx],short_acc,long_acc])

              results = np.array(results)
              plot_glushko(results,f'figures/{ID}/glushko/{step}')
    
def plot_development(results : np.array, filename : str):
    '''
    Plot the developmental trajectory of the model.
    
    Args:
        results (np.array) : array containing the magnitude of all "outputs"
                             at each timestep.
        filename (str) : base name at which to save figures. 
    '''
    
    fig = plt.figure(figsize=(15,8),dpi=400)
    x = np.linspace(1/results[-1].shape[-1],4,results[-1].shape[-1])
    
    plt.plot(x,np.mean(results[0],axis=0),label='P->P',color='blue')
    plt.plot(x,np.mean(results[1],axis=0),label='S->S',color='blue',linestyle=':')
    
    plt.plot(x,np.mean(results[2],axis=0),label='S->P',color='green')
    plt.plot(x,np.mean(results[3],axis=0),label='P->S',color='green',linestyle=':')
    
    plt.plot(x,np.mean(results[4],axis=0),label='O->P',color='orange')
    plt.plot(x,np.mean(results[5],axis=0),label='O->S',color='orange',linestyle=':')

    plt.xlabel('Timestep')
    plt.ylabel('Mean Input')

    plt.legend();
    plt.savefig(f'{filename}.png')
    plt.close('all')    

def eval_development(ckpts : List[str], model : TriangleModel, trainer : Trainer, lesions = []):
    '''
    step the model w/ a smaller timestep. :og the magnitude of
    each "output" (read: a state / hidden unit pass through
    a logistic sigmoid).
    
    Args:
        ckpts (List[str]) : list of model checkpoints to evaluate
        model (TriangleModel)
        trainer (Trainer)
    '''
    device = trainer.device

    global dataset
    for ckpt in tqdm.tqdm(ckpts[-1::],position=0):
       model.load_state_dict(torch.load(ckpt))
       model.set_lesions(lesions)
       model.eval()
        
       ID = ckpt.split('/')[1]
       step = ckpt.split('/')[-1].replace('.pth','')
       os.makedirs(f'figures/{ID}/development',exist_ok=True)        
       for data in dataset:
           with torch.no_grad():
               values = {'o2s':[],'o2p':[],'s2p':[],'p2p':[],'s2s':[],'p2s':[]}
               outputs = trainer.step(model,{'orthography':data['orthography'][None].to(device)},
                                              delta_t=1/12,return_outputs=True)[1::]
               phonology,semantics = data['phonology'][None].to(device),data['semantics'][None].to(device)            

               f = lambda x : torch.sigmoid(x)

               values['o2p'].append([model.phon_gradient.weights[2](f(output['orth_2_phon']))[phonology==1].mean().item() for output in outputs])
               values['o2s'].append([model.sem_gradient.weights[2](f(output['orth_2_sem']))[semantics==1].mean().item() for output in outputs])

               values['p2p'].append([model.phon_gradient.weights[0](f(output['cleanup_phon']))[phonology==1].mean().item() for output in outputs])
               values['s2s'].append([model.sem_gradient.weights[0](f(output['cleanup_sem']))[semantics==1].mean().item() for output in outputs])

               values['s2p'].append([model.phon_gradient.weights[1](f(output['sem_2_phon']))[phonology==1].mean().item() for output in outputs])
               values['p2s'].append([model.sem_gradient.weights[1](f(output['phon_2_sem']))[semantics==1].mean().item() for output in outputs])

       values['o2s'] = np.array(values['o2s'])
       values['o2p'] = np.array(values['o2p'])
        
       values['s2p'] = np.array(values['s2p'])
       values['p2p'] = np.array(values['p2p'])
        
       values['p2s'] = np.array(values['p2s'])
       values['s2s'] = np.array(values['s2s'])
        
       results = np.stack([values['p2s'],values['s2s'],
                           values['s2p'],values['p2p'],
                           values['o2p'],values['o2s']])
       os.makedirs(f'figures/{ID}/development/{step}/{"_".join(lesions)}',exist_ok=True)        
       plot_development(results,f'figures/{ID}/development/{step}/{"_".join(lesions)}')
    
def plot_brysbaert(results : np.array, filename : str):
    '''
    Plot and save results on Strain, et. al (1995).
    
    Args:
        results (np.array) : array containing data statistics and the 
                             model's SSE.
        filename (str) : base name at which to save figures. 
    '''
    
    fig = plt.figure(figsize=(15,8),dpi=400)
    plt.subplot(1,2,1)
    
    plt.scatter(results[:,1],results[:,2],alpha=.25)
    plt.xlabel('Concreteness')
    plt.ylabel('SSE')
    plt.title('Concreteness vs. SSE Error')
    
    normalized_frequencies = np.clip(np.sqrt(results[:,0])/(30000**.5),.05,1)
    normalized_frequencies = normalized_frequencies/np.sum(normalized_frequencies)
    
    plt.subplot(1,2,2)
    plt.scatter(normalized_frequencies,results[:,1],alpha=.25,c=results[:,2].tolist())
    
    plt.xlabel('Concreteness')
    plt.ylabel('Frequency')
    plt.title('Concreteness vs. Normalized Frequency')
    plt.savefig(f'{filename}.png')
    
    plt.close('all')
    
def eval_brysbaert(ckpts : List[str], model : TriangleModel, trainer : Trainer):
    '''
    Compare model against Brysbaert et. al (2014) concreteness ratings.
    
    Args:
        ckpts (List[str]) : list of model checkpoints to evaluate
        model (TriangleModel)
        trainer (Trainer)
    '''
    
    data = pd.read_csv('datasets/JasonLo/df_train.csv')
    orthography,phonology = data['ort'],data['pho']
    frq = data['wf']
    
    concreteness_data = pd.read_excel('datasets/JasonLo/concreteness.xlsx')
    concreteness_dict = concreteness_data[['Word','Conc.M']].set_index('Word').to_dict(orient='index')

    words = {}
    for idx,word in enumerate(data['word']):
        if concreteness_dict.get(word,None) is not None:
            words[word] = [orthography.iloc[idx],phonology.iloc[idx],frq.iloc[idx],
                           concreteness_dict[word]['Conc.M']]
            
    global orthography_tokenizer,phonology_tokenizer
    
    device = trainer.device

    ### Iterate over checkpoints
    for ckpt in tqdm.tqdm(ckpts[-1::],position=0):
       model.load_state_dict(torch.load(ckpt,map_location=device))
       model.eval()
        
       ID = ckpt.split('/')[1]
       step = ckpt.split('/')[-1].replace('.pth','')
       os.makedirs(f'figures/{ID}/brysbaert/{step}',exist_ok=True)
       with torch.no_grad():
           ### Iterate over starting timestep for SSE
           for start_step in range(2,12):
               results = []
               for idx,word in tqdm.tqdm(enumerate(words),position=0):
#                   #print(idx)
                   orthography_tensor = orthography_tokenizer(words[word][0])[None].to(device)
                   phonology_tensor = phonology_tokenizer(words[word][1])[None].to(device)
                   
                   predicted_phonology,_ = trainer.step(model,{'orthography':orthography_tensor})
                   sse = (phonology_tensor-predicted_phonology[start_step::]).pow(2).sum(dim=(1,2)).mean().item()
                   results.append([words[word][2],words[word][3],sse])
                    
               results = np.array(results)
               #print(results.shape)
               plot_brysbaert(results,f'figures/{ID}/brysbaert/{step}/{start_step}')
        
if __name__=='__main__':
   parser = argparse.ArgumentParser()

   parser.add_argument('-ID',type=str,default='')
   parser.add_argument('-model_config',type=str,default='')
   parser.add_argument('-trainer_config',type=str,default='')
    
   args = parser.parse_args()

   os.makedirs('figures',exist_ok=True)

   dataset = Monosyllabic_Dataset('datasets/JasonLo/df_train.csv',
                               'datasets/JasonLo/phonetic_features.txt',
                               'datasets/JasonLo/sem_train.npz',
                               sample = False)

   orthography_tokenizer = dataset.orthography_tokenizer
   phonology_tokenizer = dataset.phonology_tokenizer
    
   phoneme_embeddings = torch.Tensor(dataset.phonology_tokenizer.embedding_table.to_numpy())
   
   if torch.cuda.is_available(): 
      device = torch.device('cuda:0')
   else:
      device = torch.device('cpu')

   ### Load model configuration
   if args.model_config != '':
        model_config = ModelConfig.from_json(args.model_config)
   else:
        model_config = ModelConfig()
        
    ### Load trainer configuration
   if args.trainer_config != '':
        trainer_config = TrainerConfig.from_json(args.trainer_config)
   else:
        trainer_config = TrainerConfig()
    
   model = model_config.create_model().to(device)
   trainer = trainer_config.create_trainer(phoneme_embeddings).to(device)
   
   print("Plotting performance...")
   plot_performance(args.ID)
    
   ckpts = glob.glob(f'ckpts/{args.ID}/phase_2/*')
   ckpts.sort(key = lambda string: int(string.split('/')[-1].replace('.pth','')))

   print("Plotting Concreteness...")
   eval_brysbaert(ckpts,model,trainer)

   print("Plotting development...")                     
   eval_development(ckpts,model,trainer,lesions=['full'])
   eval_development(ckpts,model,trainer,lesions=['o2s'])
   eval_development(ckpts,model,trainer,lesions=['o2p'])

   print("Plotting Taraban...")                     
   eval_taraban(ckpts,model,trainer)

   print("Plotting Strain...")                     
   eval_strain(ckpts,model,trainer)

   print("Plotting Glushko...")                     
   eval_glushko(ckpts,model,trainer)
