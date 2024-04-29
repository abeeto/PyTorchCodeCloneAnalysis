import torch
from net import Discriminator, Generator
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import to_img
z_dimension = 100
batch_size = 64
D = Discriminator()
G = Generator(z_dimension)
D.load_state_dict(torch.load("./model/discriminator.pth"))
G.load_state_dict(torch.load("./model/generator.pth"))
if torch.cuda.is_available():
    D = D.cuda().eval()
    G = G.cuda().eval()
with torch.no_grad():
    z = Variable(torch.randn(batch_size, z_dimension)).cuda()
    fake_img = G(z)
    fake_img = to_img(fake_img.cpu().data)
    save_image(fake_img, "./result/fake_test.png")