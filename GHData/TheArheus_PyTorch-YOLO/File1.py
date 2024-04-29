from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D, Dense, Flatten, Add, Input, Activation
from keras.layers.merge import concatenate
from keras.optimizers import adam
from keras.regularizers import l2
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

import cv2

def ReLU(x): return Activation("relu")(x)


def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1


def poolling(x, ks, st, name):
    x = AvgPool2D((ks, ks), st, name=name)(x)
    return x


def vgg_block(x):
    x = conv(x, 64, 3, "conv1_1")
    x = ReLU(x)
    x = conv(x, 64, 3, "conv1_2")
    x = ReLU(x)
    x = poolling(x, 2, 2, "pool1_1")

    x = conv(x, 128, 3, "conv2_1")
    x = ReLU(x)
    x = conv(x, 128, 3, "conv2_2")
    x = ReLU(x)
    x = poolling(x, 2, 2, "pool2_1")

    x = conv(x, 256, 3, "conv3_1")
    x = ReLU(x)
    x = conv(x, 256, 3, "conv3_2")
    x = ReLU(x)
    x = poolling(x, 2, 2, "pool3_1")

    x = conv(x, 512, 3, "conv4_1")
    x = ReLU(x)
    x = conv(x, 512, 3, "conv4_2")
    x = ReLU(x)
    x = poolling(x, 2, 2, "pool4_1")

    x = conv(x, 256, 3, "conv4_3_CPM")
    x = ReLU(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = ReLU(x)

    return x


def stage1_block(x, num_p, branch):
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = ReLU(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = ReLU(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = ReLU(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = ReLU(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    x = ReLU(x)

    return x


def staget_block(x, num_p, stage, branch):
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = ReLU(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    x = ReLU(x)

    return x


input_shape = (None, None, 3)

stages = 6
np_branch1 = 38
np_branch2 = 19


