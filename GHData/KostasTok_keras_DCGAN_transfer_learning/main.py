import h5py
import os
import main_helper as mh
import Torch2Keras
from keras_dcgan import DCGAN

def load_model_data(generator='FloyedHub', transfer_learning=True,
                    h5_dir='dataset', imgs_dir='dataset', url=None):    
    '''
    1) To load the model set: 
    generator -> 'FloyedHub' : https://github.com/ReDeiPirati/dcgan
                 'Paper'     : https://arxiv.org/abs/1511.06434
    transfer_learning -> bool
    
    The Paper Generator provides superior results, however
    the FloyedHub one comes with pre-trained weights on the
    'Labeled Faces in the Wild Home' database. 
    
    Hence, transfer_learing requires:
    a) the FloyedHub generator
    b) to download the pre-train Torch model:
        https://www.floydhub.com/redeipirati/datasets/dcgan-300-epochs-models
        /1/netG_epoch_299.pth
    c) Rename it to 'torch_generator.pth' and place it in the 'weights' sub-dir
    
    2) To properly load the data set:
        h5_dir   -> str with the absolute of reletive path
        or          where the h5_file is stored.
        imgs_dir -> str with dir where the images are stored
        or
        url -> to data.h5 file
    !!! h5 file should always be called 'data.h5' and
        data are stored in key 'data'.
    '''
    
    #================
    # Load the model
    #================
    
    if transfer_learning:
        # If Keras trained weights file does not exists,
        # attempt to create it using pre-trained Torch model
        
        parent = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(parent,'weights/generator.h5')):
            model_loading_failed = False
        else:
            weights_path = os.path.join(parent,'weights/torch_generator.pth')
            if os.path.exists(weights_path):
                Torch2Keras.converter(weights_path)
                model_loading_failed = False
            else:
                model_loading_failed = True
    else:
        model_loading_failed = False
    
    if model_loading_failed:
        print('Could not find weights. Model not loaded.')
    else:
        dcgan = DCGAN(generator, transfer_learning)
        print('Model is loaded.')
        
        #================
        # Load the data
        #================
        
        # Attempt to load from local drive
        data = mh.load_data(h5_dir=h5_dir, imgs_dir=imgs_dir)
        
        # Else attempt to load from url
        if (data is None) and (url is not None):
            try:
                import requests
                import sys
                file_name = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    'dataset/data.pth')
                with open(file_name, "wb") as f:
                    print ('Downloading %s' % data.h5)
                    response = requests.get(url, stream=True)
                    total_length = response.headers.get('content-length')

                    if total_length is None: # no content length header
                        f.write(response.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                    for data in response.iter_content(
                                chunk_size=int(total_length/100)):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('='*done, ' '*(50-done)) )    
                        sys.stdout.flush()
                data = mh.load_data(h5_dir=h5_dir, imgs_dir=imgs_dir)
                if data is None:
                    data_loading_failed = True
                else:
                    data_loading_failed = False
            except:
                data_loading_failed = True
        else:
            data_loading_failed = False
            
        if data_loading_failed:
            print('Error will attempting to load data.')
        else:
            print('Data loaded')
            
            return dcgan, data
            
if __name__ == '__main__':
    
    dcgan, data = load_model_data()
        
    dcgan.train(data=data, epochs=100, batch_size=128)