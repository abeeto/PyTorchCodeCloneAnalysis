#------------------------------------------------------------------------------
#   Implementation of the Class-gradient-based section from the paper
#   https://arxiv.org/pdf/1312.6034v2.pdf
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch, json
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchvision.transforms import functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#------------------------------------------------------------------------------
#  Get visual numpy from torch.tensor
#------------------------------------------------------------------------------
def get_visual(X):
    X = X.contiguous()[0].detach().cpu().numpy()
    X = np.transpose(X, (1,2,0))
    X = (X - X.min()) / (X.max() - X.min())
    X = (255.0 * X).astype(np.uint8)
    return X


#------------------------------------------------------------------------------
#  Parameters
#------------------------------------------------------------------------------
N_ITER = 10000
CLASS_IDX = 278
LR = 1e-2
LAMDA = 1e-6


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Load ImageNet class indices
with open("imagenet_class_index.json", 'r') as f:
    class_ind = json.load(f)
print("Class:", class_ind[str(CLASS_IDX)][1])

# Create model
model = models.resnet18(pretrained=True)
model.cuda()
model.eval()

# Freeze trained weights
for param in model.parameters():
    param.requires_grad = False

# Prepare input
inputs = torch.zeros([1, 3, 224, 224]).cuda()
inputs.requires_grad = True

# Loop
os.makedirs("class_grad_based", exist_ok=True)
for i in tqdm(range(N_ITER), total=N_ITER):
    # Update inputs
    logits = model(inputs)
    score = logits[0, CLASS_IDX]
    score_reg = score - LAMDA*(inputs**2).sum()
    score_reg.backward()

    gradients = inputs.grad
    inputs = (inputs + LR*gradients).clone().detach()
    inputs.requires_grad = True

    # Visualize
    if i%100==0:
        print(score_reg.item())
        X = get_visual(inputs)
        cv2.imwrite(os.path.join("class_grad_based", "%.4d.png"%(i+1)), X)