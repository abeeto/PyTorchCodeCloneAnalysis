#import tensorflow as tf 
##
#with tf.Session() as sess:
#    devices = sess.list_devices()
#    print(devices)
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())