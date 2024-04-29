import torch
from sparse_utils import _dense_to_sparse, fp16_mask_correction, _get_transpose_info, _transpose_with_info
from sspade import extract_mask
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import types
import copy
import os
torch.ops.load_library(os.path.join(os.path.dirname(__file__),"build/libsputnik_ops.so"))


class sparse_linear(Function):
    @staticmethod
    def forward(ctx, x, values, row_indices, row_offsets, column_indices, m, k, transpose_info):
        ctx.save_for_backward(x, values, row_indices, row_offsets, column_indices, *transpose_info)
        ctx.m = m
        ctx.k = k
        n = x.shape[-1]
        y = torch.zeros(size=(m,n), dtype=values.dtype, device='cuda')
        torch.ops.sputnik.spmm_fp16(row_indices,
                        values.clone().contiguous(), # for some mysterious memory alignment reason this clone is mandatory 
                        row_offsets,
                        column_indices,
                        x,
                        y,
                        m,
                        n,
                        k
                        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, values, row_indices, row_offsets, column_indices, *transpose_info = ctx.saved_tensors
        n = x.shape[-1]

        grad_output = grad_output.contiguous()

        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(values).type(torch.float32)
            torch.ops.sputnik.sddmm_fp32(row_indices.type(torch.int32),
                        grad_weight,
                        row_offsets.type(torch.int32),
                        column_indices.type(torch.int32),
                        x.type(torch.float32),
                        grad_output.type(torch.float32),
                        ctx.m,
                        n,
                        ctx.k
                        )
            grad_weight = grad_weight.type(values.dtype)
        else:
            grad_weight=None

        if ctx.needs_input_grad[0]:
            row_indices_t, values_t, row_offsets_t, column_indices_t = _transpose_with_info(values, transpose_info)
            grad_x = torch.zeros_like(x)
            torch.ops.sputnik.spmm_fp16(row_indices_t,
                            values_t,
                            row_offsets_t,
                            column_indices_t,
                            grad_output,
                            grad_x,
                            ctx.k,
                            n,
                            ctx.m
                            )
        else:
            grad_x = None

        return grad_x, grad_weight, None, None, None, None, None, None


def sparse_linear_forward(self, x):
    ## assume first dimension of x is batch-size
    x_shape = copy.deepcopy(x.shape)
    x = x.view(-1, x.shape[-1]).t().contiguous() # sputnik takes batch size as column dimension
    y = sparse_linear.apply(x, self.weight, self.row_indices, self.row_offsets, self.column_indices, self.out_dim, self.in_dim, self.get_transpose_info())
    y = y.t().view(*x_shape[:-1], -1).contiguous() # reconvert batch_size to the row dimension
    if self.bias is not None:
        y = y + self.bias
    return y

def get_transpose_info(self):
    return self.row_indices_t, self.row_offsets_t, self.column_indices_t, self.perm

@torch.no_grad()
def fclayer_sparsify(module, prune_percent, mixed_precision=True):
    assert mixed_precision, "doesnt support full precision yet"
    weight = module.weight
    mask = extract_mask(weight, prune_percent)
    if mixed_precision:
        mask = fp16_mask_correction(mask)
        values_dtype = torch.float16
        column_indices_dtype = torch.int16
    else:
        raise NotImplementedError

    weight = weight * mask
    values, row_indices, row_offsets, column_indices =_dense_to_sparse(weight, weight.device)
    values = values.type(values_dtype)
    #module.register_buffer('values', values.type(values_dtype))
    module.weight = torch.nn.Parameter(values, requires_grad=True)
    module.register_buffer('row_indices', row_indices.type(torch.int32))
    module.register_buffer('row_offsets', row_offsets.type(torch.int32))
    module.register_buffer('column_indices', column_indices.type(column_indices_dtype))
    module.out_dim = weight.size(0)
    module.in_dim = weight.size(1)

    assert module.in_dim <= 32767, "sputnik column indexing is 16 bit, so keep column dimension less than 32k"
    assert module.out_dim <= 32767, "sputnik column indexing is 16 bit, so keep row dimension less than 32k"

    row_indices_t, row_offsets_t, column_indices_t, perm = _get_transpose_info(module.out_dim, module.in_dim, row_indices, row_offsets, column_indices)

    module.register_buffer('row_indices_t', row_indices_t.type(torch.int32))
    module.register_buffer('row_offsets_t', row_offsets_t.type(torch.int32))
    module.register_buffer('column_indices_t', column_indices_t.type(column_indices_dtype))
    module.register_buffer('perm', perm)

    module.forward = types.MethodType(sparse_linear_forward, module)
    module.get_transpose_info = types.MethodType(get_transpose_info, module)

    return mask

if __name__ == "__main__":
    batch_size = 4096
    hsize = 1024
    prune_percent = 50
    dummy_layer = torch.nn.Linear(hsize, hsize, bias=False)
    dummy_dense_layer = torch.nn.Linear(hsize, hsize, bias=False)
    with torch.no_grad():
        dummy_dense_layer.weight.copy_(dummy_layer.weight)
    mask = fclayer_sparsify(dummy_layer, prune_percent)
    with torch.no_grad():
        dummy_dense_layer.weight *= mask

    torch.cuda.synchronize()
    model = dummy_layer.cuda().half()
    x1 = torch.rand([batch_size, hsize], dtype=torch.float16, device='cuda')
    x2 = x1.detach().clone()

    x1.requires_grad=True
    x2.requires_grad = True
    
    y1 = model(x1)
    dummy_grad = torch.randn(y1.shape, dtype=torch.float16, device='cuda') * 0.001
    y1.backward(dummy_grad)
    torch.cuda.synchronize()

    dense_model = dummy_dense_layer.half().cuda()
    y2 = dense_model(x2)
    y2.backward(dummy_grad)

    torch.cuda.synchronize()

    sparse_grad = model.weight.grad
    dense_grad = dense_model.weight.grad[mask]

    print(f"MSE Error | weight grad {torch.mean((sparse_grad-dense_grad)**2)} | output {torch.mean((y1-y2)**2)} | input grad {torch.mean((x1.grad-x2.grad)**2)}")
    print(f"First element of | Sputnik weight grad  = {sparse_grad[0]} | Dense weight grad = {dense_grad[0]}")
    print(f"First element of | Sputnik x grad  = {x2.grad[0,0]} | Dense x grad = {x1.grad[0,0]}")


