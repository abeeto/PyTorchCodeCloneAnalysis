import torch
from torch.autograd import Variable

def reconstruction_loss(original_image, reconstructed_image):
    """Basic log loss, acts as the main loss between reconstruction and original image."""
    l = original_image * torch.log(reconstructed_image + 1e-10) + (1-original_image) * torch.log(1-reconstructed_image + 1e-10)
    l = -torch.mean(l, axis=1)
    return l

def kl_divergence_loss(mean_layer, std_layer):
    """Difference between two PDFs, acts as regularizer."""
    l = torch.exp(std_layer) + torch.square(mean_layer) - 1.0 - std_layer
    l = 0.5 * torch.sum(l, axis=1)
    return l

def network_loss(reconstruction_loss, kl_divergence_loss, alpha=1, beta=1):
    return Variable(reconstruction_loss*alpha + kl_divergence_loss*beta, requires_grad=True)

if __name__ == "__main__":

    torch.manual_seed(5)
    x =  torch.rand(1,28*28)
    y = torch.rand(1,28*28)
    print(x)
    print(y)
    print(reconstruction_loss(x, y))

