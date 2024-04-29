import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train import *

"""
Load MNIST testing images resorting to two packages datasets & transforms from torchvision.
"""
trans = transforms.Compose([transforms.ToTensor()])
testx = datasets.MNIST(root=args.dataroot, train=False, transform=trans, download=True)
view_data = [testx[i][0] for i in range(args.nrows * args.ncols)]

"""
Load the saved autoencoder model.
"""
ae_net = autoencoder(args.image_size, args.encoded_size)
ae_net.load_state_dict(torch.load(args.save_ae))
"""
Load the saved gmmn model.
"""
gmmn_net = GMMN(args.noise_size, args.encoded_size)
gmmn_net.load_state_dict(torch.load(args.save_gmmn))

plt.gray()

if args.visualize == "autoencoder":
    print("Comparision of the outputs generated by the autoencoder.")
    for i in range(args.nrows * args.ncols):
        # original images
        row = i // args.ncols
        col = i % args.ncols + 1
        ax = plt.subplot(2 * args.nrows, args.ncols, 2 * row * args.ncols + col)
        plt.imshow(view_data[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed images
        ax = plt.subplot(2 * args.nrows, args.ncols, 2 * row * args.ncols + col + args.ncols)
        img = Variable(view_data[i])
        _, _, _, decoded = ae_net(img.view(1, -1))
        plt.imshow(decoded.detach().squeeze().numpy().reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

elif args.visualize == "gmmn":
    print("Images generated by the GMMN.")

    for row in range(args.nrows):
        for col in range(args.ncols):
            ax = plt.subplot(args.nrows, args.ncols, row * args.ncols + col + 1)

            noise = torch.rand((1, args.noise_size)) * 2 - 1
            encoded = gmmn_net(Variable(noise))
            _, decoded = ae_net(encoded, index = 1)

            plt.imshow(decoded.detach().squeeze().numpy().reshape(28, 28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
plt.show()
