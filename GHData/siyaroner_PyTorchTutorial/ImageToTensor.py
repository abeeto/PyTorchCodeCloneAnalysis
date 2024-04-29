from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

kitten=Image.open("./data/kitten.png")
#kitten.show()
kitten=kitten.resize((224,224))
kitten_np=np.array(kitten)
# print(kitten_np.shape)
kitten_torch=torch.from_numpy(kitten_np)
print(kitten_torch.size())
plt.imsave("./data/kitten_cropped.png",kitten_np[:,100:224,:])