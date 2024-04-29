import torch
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

x = troch.rand(3)
print(x)

torch.cuda.is_available()

!nvidia-smi