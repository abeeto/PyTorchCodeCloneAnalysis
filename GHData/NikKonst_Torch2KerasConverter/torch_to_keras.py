import torch.legacy.nn as lnn
import numpy as np

from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import concatenate

from functools import reduce
from torch.utils.serialization import load_lua

from Torch2KerasConverter.utils import lrn, sqrt, square, mulConstant, l2Normalize


class TorchToKeras:
    inputShape = (1, 1)

    def __init__(self, shape):
        self.inputShape = shape

    def torch_to_keras(self, t7_filename, outputname=None):
        model = load_lua(t7_filename, unknown_classes=True)
        if type(model).__name__ == 'hashable_uniq_dict':
            model = model.model
        model.gradInput = None

        slist = self.lua_recursive_source(lnn.Sequential().add(model), isFirst=True)
        s = self.simplify_source(slist)
        header = '''from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import concatenate
from Torch2KerasConverter.utils import lrn, sqrt, square, mulConstant, l2Normalize
'''

        if outputname is None:
            outputname = t7_filename

        varname = outputname.replace('.t7', '').replace('.', '_').replace('-', '_')
        s = '{}\n\n{}\n{} = Model(inputs=[inp], outputs=x)\n{}.summary()\n'.format(header, s[:], varname, varname)

        with open(outputname + '.py', "w") as pyfile:
            pyfile.write(s)

        inp = Input(shape=self.inputShape)
        layers = self.lua_recursive_model(lnn.Sequential().add(model), isFirst=True, inp=inp)
        model = Model(inputs=[inp], outputs=layers)
        print(model.summary())
        model.save_weights(outputname + '.h5')

    def simplify_source(self, s):
        s = map(lambda x: '{}\n'.format(x),s)
        s = map(lambda x: x[0:],s)
        s = reduce(lambda x, y: x+y, s)
        return s

    def lua_recursive_model(self, module, prev=None, isFirst=False, inp=None):
        layers = None

        if prev is None:
            prev = inp

        for m in module.modules:
            name = type(m).__name__

            if name == 'TorchObject':
                name = m._typename.replace('cudnn.', '')
                m = m._obj

            if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM':
                if not hasattr(m, 'groups') or m.groups is None:
                    m.groups = 1

                if m.padH and m.padW is not None or 0:
                    layers = ZeroPadding2D(padding=(m.padW, m.padH))(prev)
                    prev = layers

                weights = np.transpose(m.weight.numpy()
                                       .reshape((m.nOutputPlane, m.nInputPlane, m.kW, m.kH)), (2, 3, 1, 0))
                bias = m.bias.numpy()
                layers = Convolution2D(m.nOutputPlane, (m.kW, m.kH), strides=(m.dW, m.dH),
                                       weights=[weights, bias])(prev)
            elif name == 'SpatialBatchNormalization':
                weights = m.weight.numpy()
                bias = m.bias.numpy()
                running_mean = m.running_mean.numpy()
                running_var = m.running_var.numpy()
                layers = BatchNormalization(axis=len(self.inputShape), momentum=m.momentum, epsilon=m.eps,
                                            weights=[weights, bias, running_mean, running_var])(prev)
            elif name == 'ReLU':
                layers = Activation('relu')(prev)
            elif name == 'Sequential':
                layers = self.lua_recursive_model(m, prev, isFirst=isFirst)
            elif name == 'SpatialMaxPooling':
                if m.padH and m.padW is not None or 0:
                    layers = ZeroPadding2D(padding=(m.padH, m.padW))(prev)
                    prev = layers
                layers = MaxPooling2D(pool_size=(m.kW, m.kH), strides=(m.dW, m.dH))(prev)
            elif name == 'SpatialCrossMapLRN':
                layers = Lambda(lrn, arguments={'size': m.size,
                                                     'alpha': m.alpha,
                                                     'beta': m.beta})(prev)
            elif name == 'SpatialLPPooling':
                layers = self.lua_recursive_model(m, prev, isFirst=isFirst)
            elif name == 'Square':
                layers = Lambda(square)(prev)
            elif name == 'SpatialAveragePooling':
                layers = AveragePooling2D(pool_size=(m.kW, m.kH), strides=(m.dW, m.dH))(prev)
            elif name == 'MulConstant':
                layers = Lambda(mulConstant, arguments={'const': m.constant_scalar})(prev)
            elif name == 'Sqrt':
                layers = Lambda(sqrt)(prev)
            elif name == 'Reshape' or name == 'View':
                shape = tuple()
                for size in m.size:
                    shape = (size, ) + shape
                layers = Reshape(target_shape=shape)(prev)
            elif name == 'Linear':
                weights = np.transpose(m.weight.numpy(), (1, 0))
                bias = m.bias.numpy()
                layers = Dense(m.output.shape[0], weights=[weights, bias])(prev)
            elif name == 'Normalize':
                layers = Lambda(l2Normalize, arguments={'axis': len(m.output.shape)})(prev)
            elif name == 'DepthConcat':
                concat_layers = []

                w = m.outputSize[-2]
                h = m.outputSize[-1]
                for mod in m.modules:
                    cur_layers = self.lua_recursive_model(mod, prev)

                    if h != mod.output.shape[-1] or w != mod.output.shape[-2]:
                        padH1 = int((h - mod.output.shape[-1]) / 2)
                        padH2 = int((h - mod.output.shape[-1]) / 2) + (h - mod.output.shape[-1]) % 2

                        padW1 = int((w - mod.output.shape[-2]) / 2)
                        padW2 = int((w - mod.output.shape[-2]) / 2) + (w - mod.output.shape[-2]) % 2

                        cur_layers = ZeroPadding2D(padding=((padW1, padW2), (padH1, padH2)))(cur_layers)
                    concat_layers.append(cur_layers)
                layers = concatenate(concat_layers, axis=len(self.inputShape))
            elif name == 'nn.Inception':
                layers = self.lua_recursive_model(m, prev, isFirst=isFirst)
            else:
                print(name + ' Model Layer Not Implement')

            prev = layers
            isFirst = False

        return layers

    def lua_recursive_source(self, module, xv='x', prev='x', isFirst=False):
        s = []

        if isFirst:
            prev = 'inp'

        for m in module.modules:
            name = type(m).__name__

            if name == 'TorchObject':
                name = m._typename.replace('cudnn.', '')
                m = m._obj

            if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM':
                if not hasattr(m, 'groups') or m.groups is None:
                    m.groups = 1

                if m.padH and m.padW is not None or 0:
                    s += ['{} = ZeroPadding2D(padding=({}, {}))({})'.format(xv, m.padW, m.padH, prev)]
                    prev = xv

                s += ['{} = Convolution2D({}, ({}, {}), strides=({}, {}))({})'.format(xv, m.nOutputPlane, m.kW, m.kH,
                                                                                      m.dW, m.dH, prev)]
            elif name == 'SpatialBatchNormalization':
                s += ['{} = BatchNormalization(axis={}, momentum={}, epsilon={})({})'.format(xv, len(self.inputShape),
                                                                                             m.momentum, m.eps, prev)]
            elif name == 'ReLU':
                s += ["{} = Activation('relu')({})".format(xv, prev)]
            elif name == 'Sequential':
                if isFirst:
                    s += ['inp = Input(shape={})'.format(self.inputShape)]
                s += self.lua_recursive_source(m, xv=xv, prev=xv, isFirst=isFirst)
            elif name == 'SpatialMaxPooling':
                if m.padH and m.padW is not None or 0:
                    s += ['{} = ZeroPadding2D(padding=({}, {}))({})'.format(xv, m.padH, m.padW, prev)]
                    prev = xv
                s += ['{} = MaxPooling2D(pool_size=({}, {}), strides=({}, {}))({})'.format(xv, m.kW, m.kH, m.dW, m.dH,
                                                                                           prev)]
            elif name == 'SpatialCrossMapLRN':
                s += ["{} = Lambda(lrn, arguments={{'size': {}, 'alpha': {}, 'beta': {}}})({})".format(xv,
                                                                                                       m.size,
                                                                                                       m.alpha,
                                                                                                       m.beta,
                                                                                                       prev)]
            elif name == 'SpatialLPPooling':
                s += self.lua_recursive_source(m, xv=xv, prev=prev, isFirst=isFirst)
            elif name == 'Square':
                s += ['{} = Lambda(square)({})'.format(xv, prev)]
            elif name == 'SpatialAveragePooling':
                s += ['{} = AveragePooling2D(pool_size=({}, {}), strides=({}, {}))({})'.format(xv, m.kW, m.kH, m.dW,
                                                                                               m.dH, prev)]
            elif name == 'MulConstant':
                layers = (prev)
                s += ["{} = Lambda(mulConstant, arguments={{'const': {}}})({})"
                          .format(xv, m.constant_scalar, prev)]
            elif name == 'Sqrt':
                s += ['{} = Lambda(sqrt)({})'.format(xv, prev)]
            elif name == 'Reshape' or name == 'View':
                shape = ''
                for size in m.size:
                    shape = '{}, '.format(size) + shape
                s += ['{} = Reshape(target_shape=({}))({})'.format(xv, shape, prev)]
            elif name == 'Linear':
                s += ['{} = Dense({})({})'.format(xv, m.output.shape[0], prev)]
            elif name == 'Normalize':
                s += ["{} = Lambda(l2Normalize, arguments={{'axis': {}}})({})".format(xv, len(m.output.shape), prev)]
            elif name == 'DepthConcat':
                layers = ''
                w = m.outputSize[-2]
                h = m.outputSize[-1]
                for i, mod in enumerate(m.modules):
                    cur_xv = 'inception_' + str(i)
                    s += self.lua_recursive_source(mod, xv=cur_xv, prev=xv, isFirst=isFirst)
                    if h != mod.output.shape[-1] or w != mod.output.shape[-2]:
                        padH1 = int((h - mod.output.shape[-1]) / 2)
                        padH2 = int((h - mod.output.shape[-1]) / 2) + (h - mod.output.shape[-1]) % 2

                        padW1 = int((w - mod.output.shape[-2]) / 2)
                        padW2 = int((w - mod.output.shape[-2]) / 2) + (w - mod.output.shape[-2]) % 2

                        s += ['{} = ZeroPadding2D(padding=(({}, {}), ({}, {})))({})'.format(cur_xv, padW1, padW2, padH1,
                                                                                            padH2, cur_xv)]
                    s += ['\n']
                    layers += '{}, '.format(cur_xv)
                s += ['{} = concatenate([{}], axis={})'.format(xv, layers, len(self.inputShape))]
            elif name == 'nn.Inception':
                s += ['\n', '# Inception module']
                s += self.lua_recursive_source(m, xv=xv, prev=xv, isFirst=isFirst)
            else:
                print(name + ' Not Implement')
                s += '#'

            prev = xv
            isFirst = False

        s = map(lambda x: '{}'.format(x), s)
        return s
