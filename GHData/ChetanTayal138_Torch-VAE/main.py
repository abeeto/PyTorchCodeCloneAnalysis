from model import Weights, Encoder, Decoder
from utils import latent_space
from loss import reconstruction_loss, kl_divergence_loss, network_loss
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 

def forward_propogate(INPUT_IMAGE, weights, biases):

        mean, std = Encoder(INPUT_IMAGE).encode(weights, biases)
        latent_layer = latent_space(mean, std)
        decoder_output = Decoder(latent_layer).decode(weights, biases)

        return decoder_output, mean, std

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_trainset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 32, shuffle = True)
    return trainloader
        

if __name__ == "__main__":

    trainloader = load_mnist()
    
    learning_rate = 0.0001
    epochs = 150
    batch_size = 64

    IMAGE_DIM = 28 * 28
    NN_DIM = 512
    LATENT_SPACE_DIM = 2

    ALPHA = 1 
    BETA = 1


    # input is 1 x 784

    weight_list = {
            "w1": Weights("weight_matrix_encoder_hidden", [IMAGE_DIM,NN_DIM]), # 784 x 512 
            "w2": Weights("weight_mean_hidden", [NN_DIM, LATENT_SPACE_DIM]), # 512 x 2
            "w3": Weights("weight_std_hidden", [NN_DIM, LATENT_SPACE_DIM]), # 512 x 2
            "w4": Weights("weight_matrix_decoder_hidden",[LATENT_SPACE_DIM, NN_DIM]), # 2 x 512
            "w5": Weights("weight_decoder",[NN_DIM, IMAGE_DIM]) # 512 x 784
            }

    bias_list = {
            "b1": Weights("bias_matrix_encoder_hidden", [1,NN_DIM]), # 512
            "b2": Weights("bias_mean_hidden", [1,LATENT_SPACE_DIM]), # 2
            "b3": Weights("bias_std_hidden", [1,LATENT_SPACE_DIM]), # 2
            "b4": Weights("bias_matrix_decoder_hidden",[1,NN_DIM]), # 512
            "b5": Weights("bias_decoder",[1,IMAGE_DIM]) # 784
        }
    

    

    for i in tqdm(range(1,epochs)):

        print(f"Epoch {i}")
    
        for batch, _ in trainloader:
            
            batch = batch.view(batch.shape[0], -1)
            decoder_output, mean, std = forward_propogate(batch, weight_list, bias_list)
            total_loss = ALPHA * reconstruction_loss(batch, decoder_output) + BETA * kl_divergence_loss(mean, std)
            total_loss.sum().backward()
                        
            for weight in weight_list:
                weight_list[weight]._get_weight().data = weight_list[weight]._get_weight().data - learning_rate * weight_list[weight]._get_weight().grad.data
                weight_list[weight]._get_weight().grad.data.zero_()
                
            for bias in bias_list:
                bias_list[bias]._get_weight().data = bias_list[bias]._get_weight().data - learning_rate * bias_list[bias]._get_weight().grad.data
                bias_list[bias]._get_weight().grad.data.zero_()
        
        if i % 2 == 0:
            print("Total Loss : ", total_loss.sum().data)

    # Testing    
    n = 20
    x_limit = np.linspace(-2,2,n)
    y_limit = np.linspace(-2,2,n)

    empty_image = np.empty((28*n,28*n))


    for i, zi in enumerate(x_limit):
        for j, pi in enumerate(y_limit):
            generated_latent_layer=  np.array([[zi, pi]] * batch_size)
            Decoder_Noisy = Decoder(generated_latent_layer)
            generated_image = Decoder_Noisy.decode(weight_list, bias_list)
            generated_image = generated_image.detach().numpy()
            empty_image[(n-i-1)*28 : (n-i)*28, j*28:(j+1)*28] = generated_image[0].reshape(28,28)

    plt.figure(figsize=(8,10))
    X, Y = np.meshgrid(x_limit, y_limit)
    plt.imshow(empty_image, origin="upper", cmap="gray")
    plt.grid('False')
    plt.show()



