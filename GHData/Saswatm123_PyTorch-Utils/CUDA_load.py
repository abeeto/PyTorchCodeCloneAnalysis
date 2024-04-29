import torch

def CUDA_load(dataset, transpose_data = False, dtype_X = torch.float, dtype_y = torch.float):
    '''
        Pass in iterable in form [ (X0, y0), ..., (Xn, yn) ] (transpose_data = False) 
        or [ (X0, ..., Xn), (y0, ..., yn) ] (transpose_data = True), as well as 
        data types to convert X and y to if defaults are not desired.
        
        Returns object to pass into DataLoader as 'dataset' argument. Done this way to
        allow for custom DataLoader to be used.
    '''
    if not transpose_data:
        return list([torch.tensor(X, device = 'cuda', dtype = dtype_X), torch.tensor(y, device = 'cuda', dtype = dtype_y)] for X, y in dataset)
    else:
        return list([torch.tensor(X, device = 'cuda', dtype = dtype_X), torch.tensor(y, device = 'cuda', dtype = dtype_y)] for X, y in zip(*dataset) )
