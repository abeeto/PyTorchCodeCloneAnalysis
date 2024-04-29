import torch
import json
import os
import argparse
import random
import pickle
import tqdm
import glob
import numpy as np
from dataset import Monosyllabic_Dataset
from model import ModelConfig
from train import forward_euler,TrainerConfig

os.environ['BOUND'] = '15'

class TriangleOptimizer:
    def __init__(self,model,optimizer_hyperparams):

       lr = optimizer_hyperparams.get('lr',{})
       weight_decay = optimizer_hyperparams.get('weight_decay',{})
       betas = optimizer_hyperparams.get('betas',{})
    
       opt_alg = optimizer_hyperparams.get('optimizer','SGD')
       if opt_alg == "SGD":
          opt = torch.optim.SGD
       elif opt_alg == "Adam":
          opt = torch.optim.Adam
       elif opt_alg == 'AdamW':
          opt = torch.optim.AdamW
       else:
          raise ValueError('Invalid choice of optimizer')

       ### Optimizer for the phonology (P2P) task
       self.opt_phon = opt(
                       list(model.p2p_gradient.parameters()) +\
                       list(model.phon_gradient.weights[0].parameters()),
                       lr.get('phon',5e-3),weight_decay=weight_decay.get('phon',0),
                       betas=(betas.get('phon') *.9,betas.get('phon') *.999))
       ### Optimizer for the semantics (S2S) task
       self.opt_sem = opt(
                       list(model.s2s_gradient.parameters()) +\
                       list(model.sem_gradient.weights[0].parameters()),
                       lr.get('sem',5e-3),weight_decay=weight_decay.get('sem',0),
                       betas=(betas.get('sem') *.9,betas.get('sem') *.999))

       ### Optimizer for the production (S2P + P2P) task
       self.opt_prod = opt(
                       list(model.p2p_gradient.parameters()) +\
                       list(model.s2p_gradient.parameters()) +\
                       list(model.phon_gradient.weights[0].parameters()) +\
                       list(model.phon_gradient.weights[1].parameters()),
                       lr.get('prod',5e-3),weight_decay=weight_decay.get('prod',0),
                       betas=(betas.get('prod') *.9,betas.get('prod') *.999))

       ### Optimizer for the comprehension (P2S + S2S) task
       self.opt_comp = opt(
                       list(model.s2s_gradient.parameters()) +\
                       list(model.p2s_gradient.parameters()) +\
                       list(model.sem_gradient.weights[0].parameters()) +\
                       list(model.sem_gradient.weights[1].parameters()),
                       lr.get('comp',5e-3),weight_decay=weight_decay.get('comp',0),
                       betas=(betas.get('comp') *.9,betas.get('comp') *.999))
    
       ### Optimizer for the reading (O2P + O2S) task.
       self.opt_read = opt(
                          list(model.o2s_gradient.parameters()) +\
                          list(model.o2p_gradient.parameters()) +\
                          list(model.sem_gradient.weights[2].parameters()) +\
                          list(model.sem_gradient.weights[3].parameters()) +\
                          list(model.phon_gradient.weights[2].parameters()) +\
                          list(model.phon_gradient.weights[3].parameters()),
                          lr.get('read',5e-3),weight_decay=weight_decay.get('read',0),
                          betas=(betas.get('read') *.9,betas.get('read') *.999))

    def save(self,base_path):
        torch.save(self.opt_phon.state_dict(),base_path + '_opt_phon.pth')
        torch.save(self.opt_sem.state_dict(),base_path + '_opt_sem.pth')
        torch.save(self.opt_prod.state_dict(),base_path + '_opt_prod.pth')
        torch.save(self.opt_comp.state_dict(),base_path + '_opt_comp.pth')
        torch.save(self.opt_read.state_dict(),base_path + '_opt_read.pth')

    def load(self,base_path):
        self.opt_phon.load_state_dict(torch.load(base_path + '_opt_phon.pth'))
        self.opt_sem.load_state_dict(torch.load(base_path + '_opt_sem.pth'))
        self.opt_prod.load_state_dict(torch.load(base_path + '_opt_prod.pth'))
        self.opt_comp.load_state_dict(torch.load(base_path + '_opt_comp.pth'))
        self.opt_read.load_state_dict(torch.load(base_path + '_opt_read.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ### We save ckpts,metrics,etc. under .../[ID]/...
    parser.add_argument('-ID',type=str,default='baseline')
    parser.add_argument('-ckpt_ID',type=str,default='')

    parser.add_argument('-batch_size',type=int,default=25)
    parser.add_argument('-dataset',type=str,default='jason')
    
    ### Paths to model/trainer/optimizer hyperparameters
    parser.add_argument('-model_config',type=str,default='')
    parser.add_argument('-trainer_config',type=str,default='')
    parser.add_argument('-optimizer_config',type=str,default='')
    
    ### Setting [phase_*_initial_step] > 0 will resume training
    ### at the corresponding checkpoint. [phase_*_steps] controls
    ### the # of training iterations.
    parser.add_argument('-phase_1_initial_step',type=int,default=0)
    parser.add_argument('-phase_1_steps',type=int,default=250000)
    parser.add_argument('-phase_2_initial_step',type=int,default=0)
    parser.add_argument('-phase_2_steps',type=int,default=250000)
    
    ### Sets evaluation and checkpointing intervals for training phase 1
    ### (i.e, oral learning tasks)
    parser.add_argument('-phase_1_eval_interval',type=int,default=5000)
    parser.add_argument('-phase_1_ckpt_interval',type=int,default=5000)

    ### Sets evaluation and checkpointing intervals for training phase 2
    ### (i.e, reading tasks)
    parser.add_argument('-phase_2_eval_interval',type=int,default=2500)
    parser.add_argument('-phase_2_ckpt_interval',type=int,default=5000)

    ### During phase 1 of training, freeze weights at H&S, 04 performance
    parser.add_argument('-early_stopping',action='store_true')

    ### Skip semantics and phonology tasks
    parser.add_argument('-no_cleanup',action='store_false')

    ### Detach 
    parser.add_argument('-detach',action='store_false')
    
    args = parser.parse_args()
    
    os.makedirs(f'metrics/{args.ID}',exist_ok=True)
    os.makedirs(f'ckpts/{args.ID}/phase_1',exist_ok=True)
    os.makedirs(f'ckpts/{args.ID}/phase_2',exist_ok=True)

    torch.manual_seed(0)
    random.seed(0)
    torch.autograd.set_detect_anomaly(True)

    ### Load in dataset w/ and w/o frequency sampling. The former is used 
    ### for training and the latter for evaluation.
    ### TODO: Add support for Chang et. al, 2020 data

    if args.dataset == 'jason':
       train = 'datasets/JasonLo/df_train.csv'
       phon_features = 'datasets/JasonLo/phonetic_features.txt'
       sem = 'datasets/JasonLo/sem_train.npz'
    elif args.dataset == 'chang':
       train = 'datasets/Chang2020/train.tsv'
       phon_features = 'datasets/Chang2020/features.tsv'
       sem = 'datasets/Chang2020/train_sem.npy'

    sample_dataset = Monosyllabic_Dataset(train,phon_features,sem)
    no_sample_dataset = Monosyllabic_Dataset(train,phon_features,sem,
                                   sample = False)

    sample_loader = torch.utils.data.DataLoader(sample_dataset,shuffle=True,batch_size=args.batch_size,drop_last=True)
    no_sample_loader = torch.utils.data.DataLoader(no_sample_dataset,shuffle=True,batch_size=10,drop_last=False)
    phoneme_embeddings = torch.Tensor(sample_dataset.phonology_tokenizer.embedding_table.to_numpy())
    
    ### Load model configuration
    if args.model_config:
        model_config = ModelConfig.from_json(args.model_config)
    else:
        model_config = ModelConfig()
        
    ### Load trainer configuration
    if args.trainer_config:
        trainer_config = TrainerConfig.from_json(args.trainer_config)
    else:
        trainer_config = TrainerConfig()

    if torch.cuda.is_available():
       device = torch.device('cuda:0')
    else: 
       device = torch.device('cpu')

    trainer = trainer_config.create_trainer(phoneme_embeddings).to(device)
    model = model_config.create_model().to(device)

    if args.optimizer_config:
       optimizer_hyperparams = json.load(open(args.optimizer_config,'r'))
    else:
       optimizer_hyperparams = {}

    optimizers = TriangleOptimizer(model,optimizer_hyperparams)

    if args.ckpt_ID == '':
       ckpt_ID = args.ID
    else:
        ckpt_ID = args.ckpt_ID

    if args.phase_1_initial_step:
       ckpt_path = glob.glob(f'ckpts/{ckpt_ID}/phase_1/{args.phase_1_initial_step}.pth')[0]
       model.load_state_dict(torch.load(ckpt_path))
       try:
          optimizers.load('ckpts/{ckpt_ID}/phase_1/{args.phase_1_initial_step}')
       except: pass;

    ### TODO: bundle optimizers / learning hyperparameters into Trainer
        
    ##################################### Phase 1 #####################################

    if args.phase_1_initial_step < args.phase_1_steps:
       if args.phase_1_initial_step == 0:
          p2p_acc,p2p_loss = [],[]
          s2s_acc,s2s_loss = [],[]

          p2s_acc,p2s_loss = [],[]
          s2p_acc,s2p_loss = [],[]

          eval_p2s_acc = []
          eval_s2p_acc = []
       else:
          p2p_loss = np.load(f'metrics/{args.ID}/train_p2p_loss').tolist()
          p2p_acc = np.load(f'metrics/{args.ID}/train_p2p_acc').tolist()

          s2s_loss = np.load(f'metrics/{args.ID}/train_s2s_loss').tolist()
          s2s_acc = np.load(f'metrics/{args.ID}/train_s2s_acc').tolist()

          p2s_loss = np.load(f'metrics/{args.ID}/train_p2s_loss').tolist()
          p2s_acc = np.load(f'metrics/{args.ID}/train_p2s_acc').tolist()

          s2p_loss = np.load(f'metrics/{args.ID}/train_s2p_loss').tolist()
          s2p_acc = np.load(f'metrics/{args.ID}/train_s2p_acc').tolist()

          eval_s2p_acc = np.load(f'metrics/{args.ID}/eval_s2p_acc').tolist()
          eval_p2s_acc = np.load(f'metrics/{args.ID}/eval_p2s_acc').tolist()

    last_val = True    
    skip_phon,skip_sem = False,False

    for current_step in range(args.phase_1_initial_step,args.phase_1_steps):
        if skip_phon and skip_sem: break;        

        ### Sample batch according to scaled word frequency.
        for data in sample_loader:
            break;

        ### Sample from tasks w/ prob. phonology = .2, semantics = .3,
        ### production = .2 and comprehension = .3
        if random.random() < .5 and args.no_cleanup:
           if random.random() < 2/5 and skip_phon is False:            
              ### Phonology training step
              losses,accs = trainer.train_p2p(model,optimizers.opt_phon,data)

              p2p_loss.append(losses)
              p2p_acc.append(accs)

              np.save(f'metrics/{args.ID}/train_p2p_loss',p2p_loss)
              np.save(f'metrics/{args.ID}/train_p2p_acc',p2p_acc)

           elif skip_sem is False:
              ### Semantics training step
              losses,accs = trainer.train_s2s(model,optimizers.opt_sem,data)

              s2s_loss.append(losses)
              s2s_acc.append(accs)

              np.save(f'metrics/{args.ID}/train_s2s_loss',s2s_loss)
              np.save(f'metrics/{args.ID}/train_s2s_acc',s2s_acc)

        else:
           if random.random() < 2/5 and skip_phon is False:
              ### Production training step
              loss,acc = trainer.train_s2p(model,optimizers.opt_prod,data)

              s2p_loss.append(loss)
              s2p_acc.append(acc)

              np.save(f'metrics/{args.ID}/train_s2p_loss',s2p_loss)
              np.save(f'metrics/{args.ID}/train_s2p_acc',s2p_acc)

           elif skip_sem is False:
              ### Comprehension training step
              loss,acc = trainer.train_p2s(model,optimizers.opt_comp,data)

              p2s_loss.append(loss)
              p2s_acc.append(acc)

              np.save(f'metrics/{args.ID}/train_p2s_loss',p2s_loss)
              np.save(f'metrics/{args.ID}/train_p2s_acc',p2s_acc)

        if (current_step+1)%(1e2) == 0:

            if last_val: 
               print("\n---------------------Train----------------------------\n")
               last_val = False
            else:
               print("\n------------------------------------------------------\n")

            print(current_step)

            try:
               print(np.array(np.load(f'metrics/{args.ID}/train_p2p_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            except:
               print(None)
            try:
               print(np.array(np.load(f'metrics/{args.ID}/train_s2s_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            except:
               print(None)

            print(np.array(np.load(f'metrics/{args.ID}/train_s2p_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            print(np.array(np.load(f'metrics/{args.ID}/train_p2s_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            
        if (current_step+1)%(args.phase_1_ckpt_interval) == 0:
            torch.save(model.state_dict(),f'ckpts/{args.ID}/phase_1/{current_step}.pth')
            optimizers.save(f'ckpts/{args.ID}/phase_1/{current_step}')

        ### Pass the evaluation data (again, identical to the training data, but w/o 
        ### frequency sampling) through the model. 
        if (current_step+1)%(args.phase_1_eval_interval) == 0:
            all_s2p_acc,all_p2s_acc = np.zeros((3,3)),np.zeros((3,3))
            for data in no_sample_loader:
                _,acc = trainer.train_s2p(model,None,data)

                acc = np.array(acc)
                #print(all_s2p_acc.shape,acc.shape)
                all_s2p_acc += data['phonology'].shape[0] * acc

                _,acc = trainer.train_p2s(model,None,data)

                acc = np.array(acc)
                all_p2s_acc += data['phonology'].shape[0] * acc
                
            eval_s2p_acc.append(all_s2p_acc/len(no_sample_dataset))
            eval_p2s_acc.append(all_p2s_acc/len(no_sample_dataset))

            print("\n--------------------Eval-----------------------------\n")
            print(current_step)
            print(eval_s2p_acc[-1][-1])
            print(eval_p2s_acc[-1][-1])

            if args.early_stopping:
               if eval_s2p_acc[-1][-1][0] > .9:
                  skip_phon = True
               if eval_p2s_acc[-1][-1][1] > .86:
                  skip_sem = True

            last_val = True

            np.save(f'metrics/{args.ID}/eval_s2p_acc',eval_s2p_acc)
            np.save(f'metrics/{args.ID}/eval_p2s_acc',eval_p2s_acc)
            
    ##################################### Phase 2 #####################################

    if args.phase_2_initial_step > 0:
       ckpt_path = glob.glob(f'ckpts/{args.ID}/phase_2/{args.phase_2_initial_step}.pth')[0]
       model.load_state_dict(torch.load(ckpt_path))

    o2p_loss,o2p_acc = [],[]
    o2s_loss,o2s_acc = [],[]

    o2p_only_loss,o2p_only_acc = [],[]
    o2s_only_loss,o2s_only_acc = [],[]
    
    eval_o2p_acc = []
    eval_o2s_acc = []
    eval_o2p2s_acc = []
    eval_o2s_only_acc = []
    eval_o2s2p_acc = []
    eval_o2p_only_acc = []

    last_val = True
    for current_step in range(args.phase_2_steps):

        ### Sample batch according to scaled word frequency.
        for data in sample_loader:
            break;

        ### Reading training step
        if random.random() < 10:
           losses,accs = trainer.train_full(model,optimizers.opt_read,data,detach = args.detach)

           o2p_loss.append(losses[0])
           o2p_acc.append(accs[0])

           np.save(f'metrics/{args.ID}/train_o2p_loss',o2p_loss)
           np.save(f'metrics/{args.ID}/train_o2p_acc',o2p_acc)

           o2s_loss.append(losses[1])
           o2s_acc.append(accs[1])

           np.save(f'metrics/{args.ID}/train_o2s_loss',o2s_loss)
           np.save(f'metrics/{args.ID}/train_o2s_acc',o2s_acc)

        if (current_step+1)%(1e2) == 0:
            if last_val:
                print("\n---------------------Train----------------------------\n")
                last_val = False
            else:
                print("\n------------------------------------------------------\n")

            print(current_step)

            print(np.array(np.load(f'metrics/{args.ID}/train_o2p_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            print(np.array(np.load(f'metrics/{args.ID}/train_o2s_acc.npy'))[-int(1e2)::,-1].mean(axis=0))
            
        if (current_step+1)%(args.phase_2_ckpt_interval) == 0:
            torch.save(model.state_dict(),f'ckpts/{args.ID}/phase_2/{current_step}.pth')
            optimizers.save(f'ckpts/{args.ID}/phase_2/{current_step}')

        ### Pass evaluation data through the model.
        if (current_step+1)%(args.phase_2_eval_interval) == 0 or current_step == 0:
            with torch.no_grad():
                 all_o2p_acc,all_o2s_acc = np.zeros((12,3)), np.zeros((12,3))
                 all_o2p2s_acc,all_o2s_only_acc = np.zeros((12,3)), np.zeros((12,3))
                 all_o2s2p_acc,all_o2p_only_acc = np.zeros((12,3)), np.zeros((12,3))

                 batch = 0
                 for data in no_sample_loader:
                     _,accs = trainer.train_full(model,None,data)
                     batch += data['phonology'].shape[0]

                     all_o2p_acc += data['phonology'].shape[0] * np.array(accs[0])
                     all_o2s_acc += data['semantics'].shape[0] * np.array(accs[1])

                     _,accs = trainer.train_full(model,None,data,lesions=['o2s','s2p'])

                     all_o2p2s_acc += data['semantics'].shape[0] * np.array(accs[1])
                     all_o2p_only_acc += data['phonology'].shape[0] * np.array(accs[0])

                     _,accs = trainer.train_full(model,None,data,lesions=['o2p','p2s'])
                     all_o2s2p_acc += data['phonology'].shape[0] * np.array(accs[0])
                     all_o2s_only_acc += data['semantics'].shape[0] * np.array(accs[1])
                
            eval_o2p_acc.append(all_o2p_acc/len(no_sample_dataset))
            eval_o2s_acc.append(all_o2s_acc/len(no_sample_dataset))
            eval_o2p2s_acc.append(all_o2p2s_acc/len(no_sample_dataset))
            eval_o2s_only_acc.append(all_o2s_only_acc/len(no_sample_dataset))
            eval_o2s2p_acc.append(all_o2s2p_acc/len(no_sample_dataset))
            eval_o2p_only_acc.append(all_o2p_only_acc/len(no_sample_dataset))
                
            print("\n--------------------Eval-----------------------------\n")
            print(current_step)
            print("Base Accuracies:")
            print(eval_o2p_acc[-1][-1])
            print(eval_o2s_acc[-1][-1])
            print("Indirect Accuracies:")
            print(eval_o2s2p_acc[-1][-1])
            print(eval_o2p2s_acc[-1][-1])
            print("Direct Accuracies:")
            print(eval_o2p_only_acc[-1][-1])
            print(eval_o2s_only_acc[-1][-1])

            last_val = True            

            np.save(f'metrics/{args.ID}/eval_o2p_acc',eval_o2p_acc)
            np.save(f'metrics/{args.ID}/eval_o2s_acc',eval_o2s_acc)
            np.save(f'metrics/{args.ID}/eval_o2p2s_acc',eval_o2p2s_acc)
            np.save(f'metrics/{args.ID}/eval_o2s_only_acc',eval_o2s_only_acc)
            np.save(f'metrics/{args.ID}/eval_o2s2p_acc',eval_o2s2p_acc)
            np.save(f'metrics/{args.ID}/eval_o2p_only_acc',eval_o2p_only_acc)
