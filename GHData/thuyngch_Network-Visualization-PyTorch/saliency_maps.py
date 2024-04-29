#------------------------------------------------------------------------------
#   Implementation of the Saliency map section from the paper
#   https://arxiv.org/pdf/1312.6034v2.pdf
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch, json
import numpy as np
from torchvision import models
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Create model
model = models.resnet18(pretrained=True)
model.cuda()
model.eval()

# Freeze trained weights
for param in model.parameters():
    param.requires_grad = False

# Load ImageNet class indices
with open("imagenet_class_index.json", 'r') as f:
    class_ind = json.load(f)

# Read and Pre-process an image
img = cv2.imread("images/dog.jpg")[...,::-1]
image = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
X = F.to_tensor(image)
X = F.normalize(X, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
X = torch.unsqueeze(X, dim=0)
X = X.cuda()
X.requires_grad = True

# Compute score and back-propagate to the input
logits = model(X)
idx = torch.argmax(logits, dim=1).item()
softmaxs = torch.nn.functional.softmax(logits, dim=1)
print("Predicted class:", class_ind[str(idx)][1])
print("Class score:", softmaxs[0, idx].item())
score = logits[0, idx]
score.backward()
W = X.grad.cpu().numpy()

# Post-process
W = W[0,...]
W = np.transpose(W, (1,2,0))
W = np.abs(W)
W = np.max(W, axis=2)
W = (W - W.min()) / (W.max() - W.min())

# Visualize
plt.figure()
plt.subplot(1,2,1); plt.imshow(image); plt.axis('off'); plt.title("image")
plt.subplot(1,2,2); plt.imshow(W); plt.axis('off'); plt.title("salient map")
plt.show()