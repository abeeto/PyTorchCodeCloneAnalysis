import torch

v1 = torch.tensor([1.0, 1.0], requires_grad=True) #Calculates gradients
v2 = torch.tensor([2.0,2.0]) #Doesn't calculate gradients

v_sum = v1+v2
v_res = (v_sum*2).sum()
print(v_res)

#see graph_tensor_gradient for what looks like

print(v1.is_leaf, v2.is_leaf) #Created by user? Yes so both true
print(v_sum.is_leaf, v_res.is_leaf) #Creater by user? No they are the result of a transformation so false
print(v1.requires_grad, v2.requires_grad) #True and false because one is set to true, one is false by default
print(v_sum.requires_grad, v_res.requires_grad) #Both true since v1 requires grad

v_res.backward() #Function calculates the numerical derivative of v_res wrt any var graph has
b = v1.grad
print(b) #prints out 2,2 meaning that in v1's gradients, increasing every element of v1 by one results in v_res growing by two
print(v2.grad) #as expected it's nothing since requires_grad == false
