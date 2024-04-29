import torch
from sparse_utils import _dense_to_sparse, fp16_mask_correction
from sspade import extract_mask
torch.ops.load_library("build/libsputnik_ops.so")

fp16=True
sparsity = 90
in_dim = 256
out_dim = in_dim
batch_size = 64

if fp16:
    column_indices_dtype = torch.int16
    values_dtype = torch.float16
else:
    column_indices_dtype = torch.int32
    values_dtype = torch.float16

W_dense = torch.rand((out_dim, in_dim), device='cuda', dtype=values_dtype)

## create boolean mask
mask = extract_mask(W_dense, sparsity)
if fp16:
    mask = fp16_mask_correction(extract_mask(W_dense, sparsity))

## apply mask
W_dense = W_dense * mask

with torch.no_grad():
    ## forward pass
    values, row_indices, row_offsets, column_indices =_dense_to_sparse(W_dense, 'cuda')
    values = values.type(values_dtype)
    row_indices = row_indices.type(torch.int32)
    column_indices = column_indices.type(column_indices_dtype)
    b = torch.rand(size=(W_dense.shape[-1], batch_size), dtype=values_dtype, device='cuda')
    m = W_dense.shape[0]
    n = b.shape[-1]
    k = b.shape[0]
    output = torch.zeros(size=(m,n), dtype=values_dtype, device='cuda')

    torch.ops.sputnik.spmm_fp16(row_indices,
                    values,
                    row_offsets,
                    column_indices,
                    b,
                    output,
                    m,
                    n,
                    k
                    )

    torch.cuda.synchronize()

    ## backward pass
    grad_weight = torch.zeros_like(values).type(torch.float32)
    grad_output = torch.randn_like(output).type(torch.float32)
    torch.ops.sputnik.sddmm_fp32(row_indices.type(torch.int32),
                    grad_weight,
                    row_offsets.type(torch.int32),
                    column_indices.type(torch.int32),
                    b.type(torch.float32),
                    grad_output,
                    m,
                    n,
                    k
                    )
    grad_weight = grad_weight.type(values.dtype)


W_dense.requires_grad = True
b.requires_grad = True
out_dense = torch.matmul(W_dense, b)
out_dense.backward(grad_output.type(out_dense.dtype))
grad_weight_correct = W_dense.grad[mask]

print(f"first elements of | ground truth grad = {grad_weight_correct[0]} | sparse grad = {grad_weight[0]}")
print(f"FP16 = {fp16}, out_dim={out_dim}, in_dim={in_dim}, batch_size={batch_size}, SpMM MSE={torch.mean((out_dense.float()-output.float())**2)}, SDDMM MSE={torch.mean((grad_weight-grad_weight_correct)**2)}")
