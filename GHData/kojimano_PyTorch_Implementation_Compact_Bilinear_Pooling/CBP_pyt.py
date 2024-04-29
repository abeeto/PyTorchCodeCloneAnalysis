"""
PyTorch Implemntation
"""
import numpy as np
import torch as T
import pytorch_fft.fft as fft
import pytorch_fft.fft.autograd as Fft
from torch.autograd import Variable
from IPython import embed

def _generate_sketch_matrix_pyt(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling
    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.
    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))
    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],rand_h[..., np.newaxis]), axis=1)
    indices_pyt = T.from_numpy(indices).long()
    rand_s_pyt = T.from_numpy(rand_s).float()
    sparse_sketch_matrix = T.sparse.FloatTensor(indices_pyt.t(), rand_s_pyt, T.Size([input_dim,output_dim]))
    
    return sparse_sketch_matrix


def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, not_variable=True, sum_pool=False,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=True,
    compute_size=128):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:
    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
    Args:
        bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
        sequential: (Optional) if True, use the sequential FFT and IFFT
                    instead of tf.batch_fft or tf.batch_ifft to avoid
                    out-of-memory (OOM) error.
                    Note: sequential FFT and IFFT are only available on GPU
                    Default: True.
        compute_size: (Optional) The maximum size of sub-batch to be forwarded
                      through FFT or IFFT in one time. Large compute_size may
                      be faster but can cause OOM and FFT failure. This
                      parameter is only effective when sequential == True.
                      Default: 128.
    Returns:
        Compact bilinear pooled results of shape [batch_size, output_dim] or
        [batch_size, height, width, output_dim], depending on `sum_pool`.
    """

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.size()[-1]
    input_dim2 = bottom2.size()[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix_pyt(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix_pyt(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = bottom1_pyt.view(-1, input_dim1).float()
    bottom2_flat = bottom2_pyt.view(-1, input_dim2).float()

    
   

    # Essentially:_
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    if not_variable == True:
	sketch1 = T.mm(sparse_sketch_matrix1.t().cuda(), bottom1_flat.t()).t()
	sketch2 = T.mm(sparse_sketch_matrix2.t().cuda(), bottom2_flat.t()).t()
    else:
	dense1 = Variable(sparse_sketch_matrix1.to_dense(), requires_grad = True).cuda()
	dense2 = Variable(sparse_sketch_matrix2.to_dense(), requires_grad = True).cuda()
	sketch1 = T.matmul(bottom1_flat, dense1).cuda()
	sketch2 = T.matmul(bottom2_flat, dense2).cuda()
	
    # Step 2: FFT
    if not_variable == True:
        fft1_real, fft1_img = fft.fft(sketch1, T.zeros(sketch1.size()).cuda())
        fft2_real, fft2_img  = fft.fft(sketch2, T.zeros(sketch2.size()).cuda())
    else:
	f = Fft.Fft()
        fft1_real, fft1_img = f(sketch1, Variable(T.zeros(sketch1.size())).cuda())
        fft2_real, fft2_img  = f(sketch2, Variable(T.zeros(sketch2.size())).cuda())

    # Step 3: Elementwise product
    fft_product_real = fft1_real * fft2_real -  fft1_img * fft2_img # The result of only real number part

    fft_product_img = fft1_real * fft2_img + fft2_real * fft1_img # The result of only real number part
    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    #cbp_flat = tf.real(_ifft(fft_product, sequential, compute_size))

    if not_variable == True:
        cbp_flat, _ = fft.ifft(fft_product_real, fft_product_img)	
    else:
	fi = Fft.Ifft()
        cbp_flat, _ = fi(fft_product_real, fft_product_img)	

    output_shape = T.Size([bottom1.size()[0], bottom1.size()[1],bottom1.size()[2], output_dim])
    cbp = cbp_flat.view(output_shape)

    # Step 5: Sum pool over spatial dimensions, if specified
    if sum_pool:
        cbp = T.sum(T.sum(cbp, dim=1),dim=2)

    return cbp


"""
Tensorflow Implementation
"""
import tensorflow as tf

def _fft(bottom, sequential, compute_size):
    if sequential:
        return sequential_batch_fft(bottom, compute_size)
    else:
        return tf.fft(bottom)

def _ifft(bottom, sequential, compute_size):
    if sequential:
        return sequential_batch_ifft(bottom, compute_size)
    else:
        return tf.ifft(bottom)

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling
    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.
    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

def compact_bilinear_pooling_layer_tf(bottom1, bottom2, output_dim, sum_pool=True,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=False,
    compute_size=128):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:
    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
    Args:
        bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
        sequential: (Optional) if True, use the sequential FFT and IFFT
                    instead of tf.batch_fft or tf.batch_ifft to avoid
                    out-of-memory (OOM) error.
                    Note: sequential FFT and IFFT are only available on GPU
                    Default: True.
        compute_size: (Optional) The maximum size of sub-batch to be forwarded
                      through FFT or IFFT in one time. Large compute_size may
                      be faster but can cause OOM and FFT failure. This
                      parameter is only effective when sequential == True.
                      Default: 128.
    Returns:
        Compact bilinear pooled results of shape [batch_size, output_dim] or
        [batch_size, height, width, output_dim], depending on `sum_pool`.
    """

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])
    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
        bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
        bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = _fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)),
                sequential, compute_size)
    fft2 = _fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)),
                sequential, compute_size)

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(_ifft(fft_product, sequential, compute_size))
    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # Step 5: Sum pool over spatial dimensions, if specified
    if sum_pool:
        cbp = tf.reduce_sum(cbp, reduction_indices=[1, 2])

    return cbp

if __name__ == '__main__':
   """
   Launching a quick test
   """
   bs = 1
   height = 2 
   width = 2
   input_dim1 = 2
   input_dim2 = 2
   output_dim = 2

   bottom1_npy = np.random.rand(bs,height,width,input_dim1)
   bottom2_npy = np.random.rand(bs,height,width,input_dim2)
  
    np.random.seed(3)
    rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    np.random.seed(4)
    rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    np.random.seed(5)
    rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    np.random.seed(6)
    rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1

   print("Launching a quick test for PyTorch Implementaion, bs:{}, height:{}, width:{}, input_dim1:{}, input_dim2:{}".format(bs, height,width, input_dim1, input_dim2))
   bottom1_pyt = Variable(T.from_numpy(bottom2_npy),requires_grad=True).cuda()
   bottom2_pyt = Variable(T.from_numpy(bottom2_npy),requires_grad=True).cuda()

   cbp_pyt = compact_bilinear_pooling_layer(bottom1_pyt, bottom2_pyt, output_dim, not_variable=False,rand_h_1=rand_h_1,rand_s_1=rand_s_1,rand_h_2=rand_h_2,rand_s_2=rand_s_2)
   # Autograd Test
   fake_cost =T.sum(cbp_pyt)
   fake_cost.backward()
   #print(cbp)

   print("Launching a verification test for Tensorflow Implementaion, bs:{}, height:{}, width:{}, input_dim1:{}, input_dim2:{}".format(bs, height,width, input_dim1, input_dim2))

   bottom1_tf = tf.to_float(tf.constant(bottom1_npy))
   bottom2_tf = tf.to_float(tf.constant(bottom2_npy))
   cbp = compact_bilinear_pooling_layer_tf(bottom1_tf, bottom2_tf, output_dim, sum_pool=False, rand_h_1=rand_h_1,rand_s_1=rand_s_1,rand_h_2=rand_h_2,rand_s_2=rand_s_2)   
   sess =tf.Session()
   print(sess.run(cbp))
   print(cbp_pyt)
