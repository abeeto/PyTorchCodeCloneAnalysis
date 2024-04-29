# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:24:25 2017

@author: Sampson
"""

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

def conv_BN_scale_relu(bottom, numout, kernelsize, stride, pad, bias=False, groups=1):
    conv = L.Convolution(bottom, kernel_size = kernelsize, stride = stride,
                         num_output = numout,pad = pad,bias_term = bias,
                         group = groups,
                         weight_filler=dict(type = 'msra'),
                         bias_filler = dict(type = 'constant'))

    BN = L.BatchNorm(conv,in_place=True,param=[dict(lr_mult=0,decay_mult=0),
                                              dict(lr_mult = 0, decay_mult = 0), 
                                              dict(lr_mult = 0, decay_mult = 0)])

    scale = L.Scale(BN, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place = True)
    return scale,relu
        
def ResNeXtBottleneck(bottom, in_channels, out_channels, stride, cardinality, widen_factor):
    """RexNeXt bottleneck type C
    Args:
        in_channels: input channel dimensionality
        out_channels: output channel dimensionality
        stride: conv stride. Replaces pooling layer.
        cardinality: num of convolution groups.
        widen_factor: factor to reduce the input dimensionality before convolution.
    """
    D = cardinality * out_channels // widen_factor
    scale1,relu1=conv_BN_scale_relu(bottom, D, kernelsize=1, stride=1, pad=0, bias=False)
    scale2,relu2=conv_BN_scale_relu(relu1, D, kernelsize=3, stride=stride, pad=1, bias=False, groups=cardinality)
    scale3,relu3=conv_BN_scale_relu(relu2, out_channels, kernelsize=1, stride=1, pad=0, bias=False)

    if in_channels == out_channels:
        scale0 = bottom
    else:
        scale0, relu0 = conv_BN_scale_relu(bottom, out_channels, kernelsize=1, stride=stride, pad=0, bias=False)
        
    wise=L.Eltwise(scale3, scale0, operation = P.Eltwise.SUM)
    wise_relu = L.ReLU(wise, in_place = True)
    return wise_relu

def classfication_layer(bottom, num_ouput):
    # Classification layer:  global average pool +  InnerProduct + Softmax
    _pool = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)
    _fc    = L.InnerProduct(_pool, num_output= num_ouput ,in_place=False,
                            weight_filler=dict(type='msra'), bias_term=True)
    return _fc

class ResNeXt():
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """        

        self.version = 'v1'
        self.solverfig = {'batch_size':128, 'gpunum':5, 'epoch_mult':300,'base_lr':0.1,'momentum':0.9,'weight_decay':0.0005}
        self.netfig = {'cardinality':8,'depth':29,'widen_factor':4,'baseWidth':64,}
        self.datafig = {
                'imgsize':[1,3,32,32] ,'Dataset':'cifar100', 'train_sam':50000, 'test_sam':10000, 'nlabels':100,
                # the path of datafile
                'train_file':'/data/lixinpeng/Public/cifar100_pad_mean_std_train_lmdb',
                'test_file':'/data/lixinpeng/Public/cifar100_mean_std_test_lmdb',
                'snapshot_prefix':'./dense-BC56/cifar100_dense56_v1'}
        
        self.stages = [self.netfig['baseWidth'],    #64
                       self.netfig['baseWidth']*self.netfig['widen_factor'],  #256
                       2*self.netfig['baseWidth']*self.netfig['widen_factor'],  #512
                       4*self.netfig['baseWidth']*self.netfig['widen_factor']]  #1024

    def block(self, bottom, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block_depth = (self.netfig['depth'] - 2) // 9
        block=bottom
        for bottleneck in range(block_depth):
            print(bottleneck)
            if bottleneck == 0:
                block = ResNeXtBottleneck(block, in_channels, out_channels, pool_stride, self.netfig['cardinality'], self.netfig['widen_factor'])
            else:
                block = ResNeXtBottleneck(block, out_channels, out_channels, 1, self.netfig['cardinality'], self.netfig['widen_factor'])
        return block
    
    def make_net(self, mode):
        bs = self.solverfig['batch_size']//self.solverfig['gpunum'] +1
        if mode == 'deploy':
            data = L.Input(name='data', ntop=1, shape=dict(dim = self.datafig['imgsize']))                         
        elif mode == 'train':
            data, label = L.Data(name='data', source=self.datafig['train_file'], 
                                 backend=P.Data.LMDB, batch_size=bs, ntop=2,
                                 transform_param=dict(mirror=True,crop_size=32,mean_value=[0,0,0],scale=1))
        else:
            data, label = L.Data(name='data', source=self.datafig['test_file'], 
                                 backend=P.Data.LMDB, batch_size=bs, ntop=2,
                                 transform_param=dict(mirror=True,crop_size=32,mean_value=[0,0,0],scale=1))
            
        scale, relu = conv_BN_scale_relu(data, 64, 3, 1, 1, bias=False)
        stage_1 = self.block(relu, self.stages[0], self.stages[1], 1)
        stage_2 = self.block(stage_1, self.stages[1], self.stages[2], 2)
        stage_3 = self.block(stage_2, self.stages[2], self.stages[3], 2)
        fc = classfication_layer(stage_3, self.datafig['nlabels'])

        if mode == 'deploy':
            prob = L.Softmax(fc, name='prob')
            return to_proto(prob)
        else:
            loss = L.SoftmaxWithLoss(fc, label)
            acc = L.Accuracy(fc, label)
            return to_proto(loss, acc)        

    def net2proto(self):
        name='ResNeXt_{}_{}_{}'.format(self.datafig['Dataset'], self.netfig['depth'], self.version)
        with open('./train_'+name+'.prototxt', 'w') as f:
            f.write('name:"{}"\n'.format(name))
            f.write(str(self.make_net(mode='train')))
    
        with open('./test_'+name+'.prototxt', 'w') as f:
            f.write('name:"{}"\n'.format(name))
            f.write(str(self.make_net(mode='train')))
    
        with open('./deploy_'+name+'.prototxt', 'w') as f:
            f.write('name:"{}"\n'.format(name))
            f.write(str(self.make_net(mode='train')))

    def solver2proto(self):
        name='ResNeXt_{}_{}_{}'.format(self.datafig['Dataset'], self.netfig['depth'], self.version)
        epoch = int(self.datafig['train_sam']/self.solverfig['batch_size'])+1
        max_iter = epoch * self.solverfig['epoch_mult']
        test_iter = int(self.datafig['test_sam']/self.solverfig['batch_size'])+1
        test_interval = epoch
    
        s = caffe_pb2.SolverParameter()
    
        s.train_net = './train_'+name+'.prototxt'
        s.test_net.append('./test_'+name+'.prototxt')
        s.test_interval = test_interval
        s.test_iter.append(test_iter)
    
        s.max_iter = max_iter
        s.type = 'Nesterov'
        s.display = int(epoch/5)
        # oscillation if lr is excessive, overfitting if lr is too small 
        s.base_lr =  self.solverfig['base_lr']
        s.momentum = self.solverfig['momentum']
        s.weight_decay = self.solverfig['weight_decay']
    
        s.lr_policy='multistep'
        s.gamma = 0.1
        s.stepvalue.append(int(0.5 * s.max_iter))
        s.stepvalue.append(int(0.75 * s.max_iter))
        s.stepvalue.append(int(0.9 * s.max_iter))
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        
        s.snapshot=5000
        s.snapshot_prefix='./snap'+self.version+'/'+name
        print(s)
        with open('./solver_'+name+'.prototxt', 'w') as f:
            f.write(str(s))
            
        print('ok!')
        
if __name__ == '__main__':
    # 数据处理方式：
    print('hello')
    net=ResNeXt()
    net.solver2proto()
    net.net2proto()
