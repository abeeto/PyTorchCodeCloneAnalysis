import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator

if __name__ == '__main__':
    """
    Load generator checkpoint, then generate a single image
    """
    generator = Generator()

    generator.load_state_dict(torch.load('generator.pth'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    generator = generator.to(device)
    generator.eval()

    noise = torch.randn((1, 100)).to(device)

    image = generator(noise).detach().to('cpu').numpy()

    image = np.squeeze(image, axis=(0, 1))

    plt.imshow(image)

    plt.show()