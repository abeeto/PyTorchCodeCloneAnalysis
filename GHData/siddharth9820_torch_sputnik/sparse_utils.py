import torch

def _diffsort(a):
    return torch.argsort(torch.diff(a), dim=0, descending=True)

def _nonzero_mask_to_sparse_csr_indices(mask, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(mask.shape) == 2
    index_dtype = torch.int32

    # Calculate the offset of each row.
    row_offsets = mask.sum(dim=-1, dtype=index_dtype).cumsum(dim=-1, dtype=index_dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))

    # Create the row indices and sort them.
    row_indices = _diffsort(row_offsets).to(index_dtype)

    # Extract the column indices for the nonzero values.
    column_indices = torch.where(mask)[1].to(index_dtype).contiguous()

    row_indices = row_indices.to(device)
    row_offsets = row_offsets.to(device)
    column_indices = column_indices.to(device)
    return row_indices, row_offsets, column_indices

def _dense_to_sparse(matrix, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(matrix.shape) == 2
    value_dtype = torch.float32

    # Extract the nonzero values.
    mask = matrix != 0
    values = matrix[mask].to(dtype=value_dtype, device=device)

    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(
        mask, device
    )
    return values, row_indices, row_offsets, column_indices

def _csr_to_coo(m, n, row_offsets, column_indices):
    # convert from compressed rows to uncompressed
    indices = torch.arange(m, dtype=row_offsets.dtype, device=row_offsets.device)
    row_sizes = torch.diff(row_offsets)
    row_coo = torch.repeat_interleave(indices, row_sizes.long())
    return row_coo, column_indices


def _coo_to_csr(m, n, row_indices, column_indices):
    # assumes coalesced coo
    row_offsets = row_indices.bincount(minlength=n).cumsum(0, dtype=row_indices.dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))
    return row_offsets, column_indices

def _get_transpose_info(m, n, row_indices, row_offsets, column_indices):
    # strategy:
    # - uncompress the rows to have data in COO format
    # - get permutation for stable sort of the columns to get the rows for the transposed matrix
    # - compress the new rows and return the permutation to be applied on the values

    # convert from compressed rows to uncompressed
    row_coo, _ = _csr_to_coo(m, n, row_offsets, column_indices)

    # get the permutation for the stable sort
    row_offsets_t, perm = column_indices.sort(dim=0, stable=True)
    column_indices_t = row_coo[perm]

    row_offsets_t, _ = _coo_to_csr(m, n, row_offsets_t, column_indices)
    row_indices_t = _diffsort(row_offsets_t).int()

    return row_indices_t, row_offsets_t, column_indices_t, perm

def _transpose_with_info(values, _transpose_info):
    row_indices_t, row_offsets_t, column_indices_t, perm = _transpose_info
    values_t = values[perm]
    return row_indices_t, values_t, row_offsets_t, column_indices_t


def fp16_mask_correction(mask):
    non_zeros_each_row = mask.sum(dim=1)
    for i in range(mask.shape[0]):
        if int(non_zeros_each_row[i]) % 2 != 0:
            for j in range(mask.shape[1]):
                if int(mask[i][j]) == 0:
                    mask[i][j] = 1
                    break
    non_zeros_each_row = mask.sum(dim=1)
    return mask
