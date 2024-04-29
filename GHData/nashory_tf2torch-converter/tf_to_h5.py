# export tensorflow model as hdf5 format.
import os, sys
import json
import argparse
import h5py
import numpy as np
#from collections import defaultdict
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='inception-v3', help='inception-v3 / vgg16')
args = parser.parse_args()
params = vars(args)
print  json.dumps(params, indent = 4)



## get model from checkpoint.
os.system('mkdir -p hdf5')
if params['model'] == 'inception-v3':
    h5file = h5py.File('hdf5/inception-v3.h5', "w")
    json_file = 'hdf5/inception-v3.json'
    # define graph
    data = tf.placeholder(tf.float32, (None, 299, 299, 3))
    logits, end_points = inception.inception_v3(data, 1000, is_training=False)    
    # get from ckpt
    reader = tf.train.NewCheckpointReader("checkpoint/inception_v3.ckpt")
    saved_shapes = reader.get_variable_to_shape_map()
    
## save weights.
init_op = tf.global_variables_initializer()
restore_vars = dict()

with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(init_op)

    layers = []
    for v in tf.trainable_variables():
        var_scope = v.name.split(':')[0]
        layers.append(var_scope)
        #print v.eval().shape
        #print v.eval().dtype
        h5file.create_dataset(var_scope, dtype='float32', data=v.eval())

        #restore_vars[var_scope]['shape'] = saved_shapes[var_scope]
        #print(saved_shapes)
        #if var_scope in saved_shapes:
        #    restore_vars[var_scope]['shape'] = saved_shapes[var_scope]
            
        #restore_vars['shape'].append(saved_shapes[var_scope])
        #resotre_vars['weight']
out = {}
out['layer_name'] = layers
json.dump(out, open(json_file, 'w'))
h5file.close()


#h5file = h5py.File('hdf5/inception_v3.h5', "w")
#arr = np.arange(100)
#tt = h5file.create_dataset(var_scope, data=arr)



'''
mylist = []
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    if tensor_name in saved_shapes:
        mylist.append(tensor_name)
        print(saved_shapes[tensor_name])    
        print(tensor_name)
    #if reader.has_tensor(tensor_name):
    #    print 'has tensor'
'''

'''
## run session and save weight into hdf5 format.
with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if tensor_name in saved_shapes:
            test = v.eval()
            print(test.shape)
'''






'''
#vgg = slim.nets.vgg.vgg_16()


def get_param():
    with tf.variable_scope('conv1', reuse=True) as scope_conv:
        W_conv1 = tf.get_variable('weights', shape=[5, 5, 1, 32])
        weights = W_conv1.eval()
        print(weights)
        #with open("conv1.weights.npz", "w") as outfile:
        #    np.save(outfile, weights)


# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()
#get_param_op = get_param()
# Add ops to save and restore all the variables.


# define graph.
data = tf.placeholder(tf.float32, (None, 299, 299, 3))
fake_data = tf.zeros([12, 299,299,3], tf.float32)
logits, end_points = inception.inception_v3(data, 1000, is_training=False)
vars_to_restore = tf.global_variables()
global_step = slim.get_or_create_global_step()
vars_to_restore.append(global_step)


reader = tf.train.NewCheckpointReader("pretrain/inception_v3.ckpt")
saved_shapes = reader.get_variable_to_shape_map()
print(type(saved_shapes))
'''


'''
mylist = []
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    if tensor_name in saved_shapes:
        mylist.append(tensor_name)
        print(saved_shapes[tensor_name])    
        print(tensor_name)
    #if reader.has_tensor(tensor_name):
    #    print 'has tensor'
'''

'''
with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if tensor_name in saved_shapes:
            test = v.eval()
            print(test.shape)
    '''


#print vars_to_restore

# open session.
#saver = tf.train.Saver(vars_to_restore)
#saver.restore(sess, "pretrain/incpetion_v3.ckpt")
#sess.run(end_points, feed_dict={data:fake_data})
#sess.run(end_points)
#end_points.eval()


#reader = tf.train.NewCheckpointReader("pretrain/inception_v3.ckpt")
'''
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    print tensor_name
    if reader.has_tensor(tensor_name):
        print 'has tensor'
'''
#saved_shapes = reader.get_variable_to_shape_map()

#for var in saved_shapes:
#    print(var)

'''
with tf.Session(inception) as sess:
    # define ops.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize.
    sess.run(init)
    
    # load pretrained model from ckpt file.
    #ckpt = saver.restore(sess, "pretrain/inception_v3.ckpt")
'''



'''

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

sess = tf.Session(graph=tf.get_default_graph())
#optimistic_restore(sess, "pretrain/inception_v3.ckpt")

## test vggnet.
##vgg = slim.nets.vgg




##with tf.Session() as sess:
##    print('success')


'''






# Reference code.
'''
from caffe.proto import caffe_pb2
import sy
import numpy as np
import h5py

net_param = caffe_pb2.NetParameter()
with open(sys.argv[1], 'r') as f:
  net_param.ParseFromString(f.read())

output_file = h5py.File(sys.argv[2], 'w')

for layer in net_param.layers:
  group = output_file.create_group(layer.name)
  for pos, blob in enumerate(layer.blobs):
    data = np.array(blob.data).reshape(blob.num, blob.channels, blob.height, blob.width)
    dataset = group.create_dataset('%03d' % pos, data=data)

output_file.close()
'''
