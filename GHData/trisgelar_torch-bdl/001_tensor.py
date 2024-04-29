import torch
print(torch.__version__)

tensor_array = torch.Tensor([[1,2], [4,5]])
tensor_array

tensor_uninitialized = torch.Tensor(3,3)
torch.numel(tensor_uninitialized)

tensor_initialized = torch.rand(2,3)
tensor_initialized
tensor_initialized.fill_(3)

tensor_int = torch.randn(5,3).type(torch.IntTensor)
tensor_int

tensor_long = torch.LongTensor([1.0, 2.0, 3.0])
tensor_long

tensor_byte = torch.ByteTensor([0, 261, 1, -5])
tensor_byte

tensor_ones = torch.ones(10)
tensor_ones

tensor_zeros = torch.zeros(10)
tensor_zeros

tensor_eye = torch.eye(3)
tensor_eye

non_zero = torch.nonzero(tensor_eye)
non_zero

tensor_ones_shape_eye = torch.ones_like(tensor_eye)
tensor_ones_shape_eye

new_tensor = tensor_initialized.add(4)
new_tensor

tensor_initialized.add_(5)
tensor_initialized

