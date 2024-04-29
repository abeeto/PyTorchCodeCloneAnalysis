from matplotlib import pyplot as plt

from model.discriminator import Discriminator
from model.generator import Generator

from utils.function import prepare_data, ritagliare_centro
from utils.parameters import *

discriminator = Discriminator()
discriminator.load_state_dict(torch.load("./log/discriminator.pt"))
discriminator.eval()

generator = Generator()
generator.load_state_dict(torch.load("./log/generator.pt"))
generator.eval()

accuratezza = []

criterion = torch.nn.BCELoss()

accuracy = 0.0
total = 0.0

dataloader = prepare_data("./dataset/testing/")

for data in dataloader:
    real_cpu, _ = data
    real_center_cpu = real_cpu[:, :, int(img_size / 4):int(img_size / 4) + int(img_size / 2),
                      int(img_size / 4):int(img_size / 4) + int(img_size / 2)]
    batch_size = real_cpu.size(0)
    real_cpu = real_cpu.cuda()
    real_center_cpu = real_center_cpu.cuda()
    real_cpu.to(device)
    real_center_cpu.to(device)

    input_real, input_cropped, real_center = ritagliare_centro(input_real, input_cropped, real_cpu, real_center,
                                                               real_center_cpu)

    output = discriminator(real_center)
    D_x = output.mean().item()
    print("label: 1 \t output: %f" % D_x)
    accuracy += D_x
    total += 1
    accuratezza.append(100 * accuracy / total)

    fake = generator(input_cropped)
    output = discriminator(fake.detach())
    D_G_z = output.mean().item()
    print("label: 0 \t output: %f" % D_G_z)
    accuracy += (1 - D_G_z)
    total += 1
    accuratezza.append(100 * accuracy / total)

print(f"Accuratezza: {round((100 * accuracy /total), 2)}%")

plt.figure()
plt.title("Accuratezza del discriminatore")
plt.plot(accuratezza)
plt.xlabel("Batch analizzati")
plt.ylabel("Accuratezza")
fig = plt.gcf()
fig.savefig('./log/accuratezza_discriminatore.png')
