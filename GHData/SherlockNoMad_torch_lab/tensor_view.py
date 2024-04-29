import torch
import pytest


def handle_neg_dim(rank, dim):
    assert -rank <= dim and dim <= rank-1, \
        "Dimension out of range, expect [{}, {}], but got {}".format(-rank, rank-1, dim)
    return (rank + dim) % rank


def lab_permute(input: torch.Tensor, dims):
    rank = input.dim()

    assert len(dims) == len(set(dims)), "repeated dim in permute"

    dims = [handle_neg_dim(rank, dim) for dim in dims]

    assert sorted(dims) == list(range(rank)), "Invalid dims"
    
    rank = len(dims)
    shape = input.size()
    stride = input.stride()

    new_shape, new_stride = list(), list()
    for dim in dims:
        new_shape.append(shape[dim])
        new_stride.append(stride[dim])

    return input.as_strided(new_shape, new_stride)


def lab_diagonal(input: torch.Tensor, offset=0, dim1=0, dim2=1) -> torch.Tensor:
    rank = input.dim()

    dim1 = handle_neg_dim(rank, dim1)
    dim2 = handle_neg_dim(rank, dim2)

    assert dim1 != dim2, "diagonal dimensions cannot be identical {}, {}".format(dim1, dim2)

    # shuffle dimensions, such that diag is always taken on last 2 dimensions
    perm = [dim for dim in range(rank) if dim not in [dim1, dim2]]
    perm.extend([dim1, dim2])

    if (perm != list(range(rank))):
        temp = lab_permute(input, perm)
    else:
        temp = input

    shape = temp.size()
    stride = temp.stride()
    
    if offset >= 0:
        diag_size = max(0, min(shape[-2], shape[-1] - offset))
        delta_offset = offset * stride[-1]
    else:
        diag_size = max(0, min(shape[-2] + offset, shape[-1]))
        delta_offset = -offset * stride[-2]

    new_shape = tuple(list(shape[:-2]) + [diag_size])
    new_stride = tuple(list(stride[:-2]) + [stride[-1] + stride[-2]])
    new_storage_offset = temp.storage_offset() + delta_offset

    output = temp.as_strided(new_shape, new_stride, storage_offset=new_storage_offset)
    return output

def lab_movedim(input: torch.Tensor, source, destination) -> torch.Tensor:
    rank = input.dim()
    perm = list(range(rank))

    if type(source) is int:
        source = (source, )
    if type(destination) is int:
        destination = (destination, )

    assert len(source) == len(destination), \
        "Invalid source or destination dims: source ({} dims ) should contain the same " \
        "number of dims as destination ({} dims)".format(source, destination)

    assert len(source) == len(set(source)), "repeated dim in `soruce` ({})".format(source)
    assert len(destination) == len(set(destination)), "repeated dim in `destination` ({})".format(destination)

    source_touched = [handle_neg_dim(rank, dim) for dim in source]
    destination_touched = [handle_neg_dim(rank, dim) for dim in destination]

    assert len(source_touched) == len(set(source_touched)), "movedim: repeated dim in `source` ({})".format(source)
    assert len(destination_touched) == len(set(destination_touched)), "movedim: repeated dim in `destination` ({})".format(destination)

    remaining_source = [dim for dim in range(rank) if dim not in source_touched]
    remaining_dest = [dim for dim in range(rank) if dim not in destination_touched]

    source_touched.extend(remaining_source)
    destination_touched.extend(remaining_dest)

    perm = [None] * rank
    for i in range(rank):
        perm[destination_touched[i]] = source_touched[i]

    return lab_permute(input, perm) 


def lab_chunk(input: torch.Tensor, chunks, dim=0) -> List[torch.Tensor]:
    
    return


x = torch.arange(120).view(2, 3, 4, 5)

x1 = x[:,2,:,:]

x2 = x1.permute(2, 0, 1)

# dims = [2,0,-2]
# y1 = torch.permute(x1, dims=dims)
# y2 = lab_permute(x1, dims)
# assert torch.allclose(y1, y2)

# y1 = lab_diagonal(x1, dim1=2, dim2=-3, offset=3)
# y2 = torch.diagonal(x1, dim1=2, dim2=-3, offset=3)
# assert torch.allclose(y1, y2)

y1 = lab_movedim(x, source=(0,-1), destination=(2,-3))
y2 = torch.movedim(x, source=(0,-1), destination=(2,-3))
assert torch.allclose(y1, y2)