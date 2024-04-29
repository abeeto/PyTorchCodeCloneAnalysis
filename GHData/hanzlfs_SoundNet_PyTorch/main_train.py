### main script for training the whole dataset

from torch.legacy import nn
from torch.utils.serialization import load_lua
import torch
opt = {
  'dataset' : 'audio',    # indicates what dataset load to use (in data.lua)
  'nThreads' : 40,        # how many threads to pre-fetch data
  'batchSize' : 64,       # self-explanatory
  'loadSize' : 22050*20,  # when loading images, resize first to this size
  'fineSize' : 22050*20,  # crop this size from the loaded image 
  'lr' : 0.001,           # learning rate
  'lambda' : 250,
  'beta1' : 0.9,          # momentum term for adam
  'meanIter' : 0,         # how many iterations to retrieve for mean estimation
  'saveIter' : 5000,      # write check point on this interval
  'niter' : 10000,        # number of iterations through dataset
  'ntrain' : float('inf'),   # how big one epoch should be
  'gpu' : 0,              # which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  'cudnn' : 1,            # whether to use cudnn or not
  'finetune' : '',        # if set, will load this network instead of starting from scratch
  'name' : 'soundnet',    # the name of the experiment
  'randomize' : 1,        # whether to shuffle the data file or not
  'data_root' : '/data/vision/torralba/crossmodal/flickr_videos/soundnet/mp3',
  'label_binary_file' : '/data/vision/torralba/crossmodal/soundnet/features/VGG16_IMNET_TRAIN_B%04d/prob',
  'label2_binary_file' : '/data/vision/torralba/crossmodal/soundnet/features/VGG16_PLACES2_TRAIN_B%04d/prob',
  'label_text_file' : '/data/vision/torralba/crossmodal/soundnet/lmdbs/train_frames4_%04d.txt',
  'label_dim' : 1000,
  'label2_dim' : 401,
  'label_time_steps' : 4,
  'video_frame_time' : 5,  # 5 seconds
  'sample_rate' : 22050,
  'mean' : 0,
}

torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_default_tensor_type('torch.FloatTensor')

#### Create data loader

##### create net work
## initialize the model
def weights_init(layer):
    name = torch.typename(layer)
    if name.find('Convolution') > 0 :
        layer.weight.normal_(0.0, 0.01)
        layer.bias.fill_(0)
        #print name, name.find('Convolution')
    elif name.find('BatchNormalization') > 0:
        if layer.weight is not None:
            layer.weight.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.fill_(0)

## create network
def create_network():
    net = nn.Sequential()
    
    net.add(nn.SpatialConvolution(1, 16, 1,64, 1,2, 0, 32))
    net.add(nn.SpatialBatchNormalization(16))
    net.add(nn.ReLU(True))
    net.add(nn.SpatialMaxPooling(1,8, 1,8))

    net.add(nn.SpatialConvolution(16, 32, 1,32, 1,2, 0, 16))
    net.add(nn.SpatialBatchNormalization(32))
    net.add(nn.ReLU(True))
    net.add(nn.SpatialMaxPooling(1,8, 1,8))

    net.add(nn.SpatialConvolution(32, 64,  1,16, 1,2, 0, 8))
    net.add(nn.SpatialBatchNormalization(64))
    net.add(nn.ReLU(True))

    net.add(nn.SpatialConvolution(64, 128, 1,8, 1,2, 0, 4))
    net.add(nn.SpatialBatchNormalization(128))
    net.add(nn.ReLU(True))
    
    net.add(nn.SpatialConvolution(128, 256, 1, 4, 1, 2, 0, 2))
    net.add(nn.SpatialBatchNormalization(256))
    net.add(nn.ReLU(True))
    
    net.add(nn.SpatialMaxPooling(1,4, 1,4))
    
    net.add(nn.SpatialConvolution(256, 512,  1,4, 1,2, 0,2))
    net.add(nn.SpatialBatchNormalization(512))
    net.add(nn.ReLU(True))
    
    net.add(nn.SpatialConvolution(512, 1024,  1,4, 1,2, 0,2))
    net.add(nn.SpatialBatchNormalization(1024))
    net.add(nn.ReLU(True))


    net.add(nn.ConcatTable().add(nn.SpatialConvolution(1024, 1000, 1,8, 1,2, 0,0))
                            .add(nn.SpatialConvolution(1024,  401, 1,8, 1,2, 0,0)))

    net.add(nn.ParallelTable().add(nn.SplitTable(3)).add(nn.SplitTable(3)))
    net.add(nn.FlattenTable())
    return net
