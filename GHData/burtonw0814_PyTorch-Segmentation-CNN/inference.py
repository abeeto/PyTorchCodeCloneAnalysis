import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile

from tifffile import imsave

from graph_class import *
from get_batch import *
from get_inst import *
from batch_queue_functions import *

from data_paths import *



'''
from eval_functions import *
'''



from get_eval_batch import *




if __name__ == '__main__': 



    

    ###########################
    num_inst=8;
    num_training_steps=200000
    summ_step=10000;
    save_step=1000; 

    # Settings for CNN
    net_number=0
    input_depth=1;
    batch_size=8;
    num_classes=9;
    pcx=256;
    pcy=256;
    train_mode=True
    ##########################





    ##################################
    ############### TEST ############
    ##################################
    data_stuff=Data_Paths(pre_cached=False, train_mode=False)

    model_path=('./models/_model_.pth')
    mod=graph_class(input_depth, num_classes, 
                        pcx, pcy, summ_step, 
                        save_step,  model_path, 
                        cuda=1, load=True, TB_ID=0)

    num_pools=data_stuff.num_pools
    print(str(num_pools) + ' evaluation directories')    
    
    #my_eval = Eval_Obj(num_classes, net_number, train_type_list[i])
    for pp in range(num_pools):
        num_eval=data_stuff.num_inst_sup[pp]
        print(data_stuff.sup_pools[pp])
        print(str(num_eval) + ' evaluation instances')     
        
        imd_block=[];
        pxd_block=[];
        
        # Predict each slice
        for ii in range(num_eval):

            if ii%10==0:
                print(ii)
            
            # Get eval instance
            b=get_eval_inst(data_stuff,
                pcx, pcy, num_classes,
                pp, ii)

            # Inference
            pred_list=mod.get_prediction(b); 

            #ov=return_overlay(b[1], pred_list)
            #plt.imshow(ov)
            #plt.show()

            # Store seg map
            imd_block.append(b[1])
            pxd_block.append(pred_list)   
        
        # 3D connected components
        pxd_block_new=connected_components(pxd_block, num_classes)
        
        
        
        # Try to make paths
        path_make_list=(data_stuff.top_path2,
                        data_stuff.top_path2 + '/' + data_stuff.subj_num[pp] + '/',
                        data_stuff.top_path2 + '/' + data_stuff.subj_num[pp] + '/' + data_stuff.scan_num[pp] + '/',
                        data_stuff.inf_out_pools[pp] + '/Ov/')#,  
                        #data_stuff.inf_out_pools[pp] + '/Pxd/',  
                        #)
        
        
        
        for jj in range(len(path_make_list)):
            try:
                os.mkdir(path_make_list[jj])
            except:
                [];#os.mkdir(path_make_list[jj])



        print("Exporting")
        for i in range(len(pxd_block_new)):
            # Get overlays
            ov=return_overlay(imd_block[i], pxd_block_new[i])
            
            ID=str(i)
            while len(ID)<6:
                ID='0'+ID

            # if i==120:
            #    plt.imshow(ov); plt.show();

            # Export
            path_out= data_stuff.inf_out_pools[pp] + '/Ov/' + ID + '.png';
            imageio.imwrite(path_out, ov)

            # 
            # path_out_seg=data_stuff.inf_out_pools[pp] +  '/Pxd/' + ID + '.png';
            # imageio.imwrite(path_out_seg, pxd_block_new[i])
            
        # Export full tiff stack
        path_out_tiffs=data_stuff.inf_out_pools[pp] + "/tiff_block.tif";
        imsave(path_out_tiffs, np.stack(pxd_block_new, axis=0).astype(np.uint8))

        # Copy tag file
        src=data_stuff.sup_pools[pp] + '/Scan_Path.txt';
        dst=data_stuff.inf_out_pools[pp] + '/Scan_Path.txt';
        copyfile(src, dst)
















# for ii in range(num_eval):
'''
plt.imshow(ov)
plt.show()
'''

# Create overlay

# Export overlay

'''
# Num sups
'''
''' EXPORT AT ORIGINAL RESOLUTION!!!!!!!!! '''
'''ground_list=b[4]

my_eval.aggregate(ground_list, pred_list)
my_eval.export(pred_list, ID_list, data_root_list)'''
''' EXPORT AT ORIGINAL RESOLUTION!!!!!!!!! '''


            


# Export predictions

# Get full IoU Vals
# Export IoU Vals

# How to eval over synthetic data

# Add path and id to prediction
# Fix IoU computation
# Export IoU vals
# Export overlatys and segmentation map predictions

# Need path so we know where to export seg maps     
# Each data set
# Cycle through each image
# Get iou, classwise
# Export segmentation maps and overlays, classwise
# Primarily need iou and segmentation maps
# Synthetic, dont care about it














