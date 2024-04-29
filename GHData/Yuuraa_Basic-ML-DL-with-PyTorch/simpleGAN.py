import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# GANs are very sensitive to hyperparameters!!
# Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3. Differente learning rate (is there a better one)?
# 4. Try changing architecture to a CNN

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1), # Slope 0.1. GAN에서는 주로 LeakyReLU를 사용
            nn.Linear(128, 1),
            nn.Sigmoid(), # Fake or real, 0 or 1
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    # z_dim은 "latent noise" (or just noise)
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28x28x1의 MNIST 데이터 -> 784로 flatten 할 것임
            nn.Tanh(), # Normalize the input from MNIST datset => -1 과 1 사이로
        )
    
    def forward(self, x):
        return self.gen(x)

# Hyperparmeters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # Important! 바꾸면서 어떤 것이 가장 나은 지 확인 할 것
z_dim = 64 # 이것도 128, 256 등 시도 가능
image_dim = 28 * 28 * 1 # 784
batch_size = 32 # Standard
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))] # Mean and standard deviation of MNIST dataset
    [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))] # Mean and standard deviation of MNIST dataset
)
dataset = dsets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake") # Output fake images (generated)
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator - Maximize log(D(real)) + log(1 - D(G(z))) z is random noize input to the Generator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1) # Flatten log(D(real)) 부분
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # minimum of log(D(real))
        # disc_fake = disc(fake.detach()).view(-1) # detach를 함으로써 backward를 할 때 정보가 사라지지 않음. 혹은, detach를 지우고
        disc_fake = disc(fake).view(-1) # detach를 함으로써 backward를 할 때 정보가 사라지지 않음. 혹은, detach를 지우고
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # 0들을 보내고 있기 때문에, log(1-D(g(z)))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True) # Forward pass에서 이용된 것은 cleared 됨. 아래에서 fake를 다시 쓰고 싶으면 어떻게 해야할까? fake에 detach()를 쓰거나, 여기에 retain_graph를 추가하면 된다
        opt_disc.step()

        ### Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
