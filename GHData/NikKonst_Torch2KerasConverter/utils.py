import tensorflow as tf
from keras import backend as K


def lrn(a, size, alpha, beta):
    return tf.nn.lrn(a, depth_radius=size, alpha=alpha, beta=beta)


def square(a):
    return a**2


def mulConstant(a, const):
    return a * const


def sqrt(a):
    return K.sqrt(a)


def l2Normalize(a, axis):
    return K.l2_normalize(a, axis=axis)
