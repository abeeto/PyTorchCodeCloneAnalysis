import numpy as np
import cv2
import scipy
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import time

from data_paths import *
from get_batch import *







def test_workflow():

    batch_size=4
    input_depth=1
    pc_y=256; pcy=pc_y;
    pc_x=256; pcx=pc_x;
    num_classes=9

    data_stuff=Data_Paths(pre_cached=False)
    
    TOT_LIST = (input_depth,
                    batch_size, 
                    pc_x, pc_y, 
                    num_classes,
                    data_stuff)

    # Retrieve batch
    t=time.time()
    b=get_batch(TOT_LIST)
    print('BATCH GEN TIME: ' + str(time.time()-t))

    bx=b[0]
    by=b[1]
    im_list=b[2]
    pxd_list=b[3]

    print(np.unique(bx[0,0,:,:]))
    #print(np.sum(bx[0,0,:,:]-bx[0,1,:,:]))
    plt.imshow(bx[0,0,:,:]); plt.show()


    #plt.imshow(im_list[0]); plt.show()
    #plt.imshow(pxd_list[0][:,:]); plt.show()
    #plt.imshow(by[0,0,:,:]); plt.show()

    plt.imshow((by[0,0,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,1,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,2,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,3,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,4,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,5,:,:]*255).astype(np.uint8)); plt.show()
    plt.imshow((by[0,6,:,:]*255).astype(np.uint8)); plt.show()

    '''
    items_predict=decode_all(im_list, by, pcx,  pcy, num_classes)
    filtered=filter_items(items_predict, num_classes)
    pred_list=create_prediction_objects(im_list,  by, filtered, pcx, pcy)
    '''



    return

















for i in range(1):
    
    test_workflow()
























