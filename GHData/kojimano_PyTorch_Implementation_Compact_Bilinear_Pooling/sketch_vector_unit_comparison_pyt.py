import tensorflow as tf
import numpy as np
from IPython import embed
output_dim = 3
input_dim2 =3
#bottom2 = tf.random_normal([8, 2000,300,3], mean=-1, stddev=4)
bottom2_npy = np.random.rand(1,2,2,3)
bottom2_tf = tf.constant(bottom2_npy)
bottom2_flat_tf = tf.to_float(tf.reshape(bottom2_tf, [-1, input_dim2]))

np.random.seed(2)
rand_h = np.random.randint(output_dim, size=input_dim2)
np.random.seed(3)
rand_s = 2*np.random.randint(2, size=input_dim2) - 1 

input_dim = len(rand_h)
indices = np.concatenate((np.arange(input_dim)[..., np.newaxis], rand_h[..., np.newaxis]), axis=1)

# Tensorflow: Step1
sparse_sketch_matrix_tf = tf.sparse_reorder(tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
dense_tf = tf.sparse_tensor_to_dense(sparse_sketch_matrix_tf)
sess = tf.Session()
print(sess.run(sparse_sketch_matrix_tf))
print(sess.run(dense_tf))

#Tensorflow: Step2
sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(tf.to_float(sparse_sketch_matrix_tf) ,bottom2_flat_tf, adjoint_a=True, adjoint_b=True))
print(sess.run(sketch2))
embed()

# Torch: Step1
import torch as T
from torch.autograd import Variable
import pytorch_fft.fft as fft
import pytorch_fft.fft.autograd as Fft
bottom2_pyt = T.from_numpy(bottom2_npy)
bottom2_flat_pyt = Variable(bottom2_pyt.view(-1, input_dim2).float(), requires_grad = True)

indices_pyt = T.from_numpy(indices).long() 
rand_s_pyt = T.from_numpy(rand_s).float()  
sparse_sketch_matrix = T.sparse.FloatTensor(indices_pyt.t(), rand_s_pyt, T.Size([input_dim,output_dim]))
dense = Variable(sparse_sketch_matrix.to_dense(), requires_grad = True)
print(sparse_sketch_matrix)
#print(dense)

# Torch: Step2
from torch.autograd import Variable
#asketch2 = T.mm(sparse_sketch_matrix.t(), bottom2_flat_pyt.t()).t()
sketch2 = T.matmul(bottom2_flat_pyt, dense)
out = sketch2.mean()
embed()
out.backward()
