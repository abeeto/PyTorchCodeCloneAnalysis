# credit: https://pytorch.org/docs/master/torch.html#torch.where

a = np.array([0.1, 0.4, 0.35, 0.8])
a = np.array([a,a])
prob = torch.from_numpy(a)  # convert numpy to torch tensor

t0 = torch.zeros(prob.shape) # create a tensor of all zeros with shape as 'prob': another approach: t0 = torch.zeros_like(prob)
t1 = torch.ones(prob.shape) # create a tensor of all ones with shape as 'prob' : : another approach: t1 = torch.ones_like(prob)
th = 0.4
out = torch.where(prob >= th, t1, t0) # binarize based on the condition
print(out) # print binarized Tensor

# convert that into boolean numpy, and 
out = out.numpy().astype('bool')

# but back to Tenso-type again isn't supported
# print(torch.from_numpy(out)) produces following error:
# TypeError: can't convert np.ndarray of type numpy.bool_. The only supported types are: double, float, float16, int64, int32, and uint8.


