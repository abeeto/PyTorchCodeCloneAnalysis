import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def plot_imgs(images, epoch, batch_size=32, rows=6, cols=6):
    images = images.squeeze() # 32x64x64
    trans = ToPILImage()

    for i in range(batch_size):
        img = images[i,:,:].cpu().detach() * 0.5 + 0.5
        img = trans(img)

        plt.subplot(rows, cols, i + 1) # 1 ~ 4
        plt.imshow(img)
        plt.axis('off')
    # plt.show()
    plt.savefig('out_imgs/fix_fake_e{:02d}.png'.format(epoch))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
