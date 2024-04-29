import time
import numpy as np
import torch
import argparse

from graph_class import *
from get_batch import *
from get_inst import *
from batch_queue_functions import *

from data_paths import *


#CUDA_VISIBLE_DEVICES='' python3 -m tensorboard.main --logdir=/home/will/Desktop/Flinders_Seg/V3/runs
if __name__ == '__main__': 

    ###########################
    num_inst=8;
    num_training_steps=300000
    summ_step=10000;
    save_step=1000; 

    # Settings for CNN
    net_number=0
    input_depth=1;
    batch_size=12;
    num_classes=9;
    pcx=256;
    pcy=256;
    train_mode=True
    ##########################

    ##################################
    ############ TRAIN ###############
    ##################################
    if train_mode==True:
        data_stuff=Data_Paths(pre_cached=False)

        # Initialize CNN1
        model_path=('./models/_model_.pth')
        mod=graph_class(input_depth, num_classes, 
                        pcx, pcy, summ_step, 
                        save_step,  model_path, 
                        cuda=1, load=True, TB_ID=0)
        
        # Initialize batch getter
        TOT_LIST = (input_depth,
                    batch_size, 
                    pcx, pcy, 
                    num_classes,
                    data_stuff)
        I=async_batch_queue(num_inst, TOT_LIST, get_batch)   
        
        # Training steps
        for i in range(num_training_steps): 
            
            # Kick off asynchronous batch processing
            I.apply_batches(TOT_LIST)
            
            # Sample a batch from instance queue
            b=I.sample_batch()
            
            # Step model
            l=mod.step_model(b) # step model using sampled batch    
            
            I.retrieve_batches()
            
            if i%(50000)==0 and i>0:
                data_stuff.update_paths()
        mod.save_model()
        mod=None; torch.cuda.empty_cache() 
        I.close_batch_queue(); I=None; time.sleep(10)

    ###############################################################
    ###############################################################
    ###############################################################
    ###############################################################
        
        
        
        
        
        
        
        
        
        
    ##################################
    ############### TEST ############
    ##################################
    if False: #test_mode==True:
        train_type_list=(0,1,2)
        eval_type_list=('synth','real') 
        for i in range(len(train_type_list)):       
            for j in range(len(eval_type_list)):

                data_stuff=Data_Paths(pre_cached=False, 
                                      train_mode=False, 
                                      eval_type=eval_type_list[j])

                model_path=('./models/_model_' + str(train_type_list[i]) + '.pth')
                mod=graph_class(input_depth, num_classes, 
                                pcx, pcy, summ_step, 
                                save_step,  model_path, 
                                cuda=1, load=True)

                # Num sups
                num_pools=self.num_pools
                print(str(num_pools) + ' evaluation directories')                
                my_eval = Eval_Obj(num_classes, net_number, train_type_list[i])

                for pp in range(1):#num_pools):
                    num_eval=data_stuff.num_inst_sup[pp]
                    print(str(num_eval) + ' evaluation instances')                    
                    for ii in range(1): #num_eval):
                        b=get_eval_inst(data_stuff,
                                    pcx, pcy, num_classes,
                                    mask_colors, pp, ii)
                        ''' EXPORT AT ORIGINAL RESOLUTION!!!!!!!!! '''
                        ground_list=b[4]
                        pred_list=mod.get_prediction(b); 
                        my_eval.aggregate(ground_list, pred_list)
                        my_eval.export(pred_list, ID_list, data_root_list)
                        ''' EXPORT AT ORIGINAL RESOLUTION!!!!!!!!! '''













