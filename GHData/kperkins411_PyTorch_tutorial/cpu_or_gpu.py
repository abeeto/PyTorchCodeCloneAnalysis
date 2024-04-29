import torch
# let us run this cell only if CUDA is available
if torch.cuda.is_available():

    # creates a LongTensor and transfers it
    # to GPU as torch.cuda.LongTensor
    a = torch.full((10,), 3, device=torch.device("cuda"))
    print(type(a))
    b = a.to(torch.device("cpu"))
    # transfers it to CPU, back to
    # being a torch.LongTensor
    print(type(b))
else:
    print("cuda not available")
