import torch

def linearKernel(yi, xi, sigma):
    # instead of comparing the output values with zero, we pass them 
    # through a relu activation function
    return torch.nn.functional.relu(1-(yi-xi).norm(dim=-1) / sigma)

def repulsive_loss(xk, xl):
    return 1/(xk-xl).norm()

def attractive_loss(xk):
    return xk.norm().square() 

def total_loss(kerns, device):

    loss = torch.zeros((1,1), device=device)

    for i in range(kerns.shape[0]):
        att_loss = attractive_loss(kerns[i][:])
        loss = loss + att_loss
        for j in range(kerns.shape[0]):
            if j != i:
                rep_loss = repulsive_loss(kerns[i][:], kerns[j][:])
                loss += rep_loss

    return loss
    

def initializeRigitKernels(numKernels, numItterations=100, device="cpu"):
    # solving an optimization problem to position the kernels

    #setting learning rate
    lr = 0.1

    # positioning a point at the origin
    origin = torch.tensor([0,0,0], device=device).unsqueeze(0)
    
    # positioning the other kernels at random points
    kernels = torch.rand((numKernels-1, 3), requires_grad=True, device=device)

    # Trainning with the original learning rate for the 90% percent of the itterations
    for i in range(numItterations):
        #print("itteration : ", i)
        
        # We want the points to be as far as possible inside a given sphere
        kerns = torch.cat([origin, kernels])

        # computing the loss
        loss = total_loss(kerns, device)
        loss.backward()
        
        # updating the kernel positions
        kernels.data = kernels - lr * kernels.grad

        # reseting the gradient
        kernels.grad = torch.zeros_like(kernels.grad)
    
    # The kernels no more require grad computation for the rigit kernel
    kernels.requires_grad=False
    return torch.cat([origin, kernels])
