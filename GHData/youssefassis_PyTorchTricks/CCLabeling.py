import torch
import torch.nn.functional as F

def getCC(img, max_iterations=100, device = 'cpu'):
    comp = img.clone()
    comp = torch.arange(comp.numel()).reshape(comp.shape).to(device).float()
    comp[img != 1] = 0
    count = 0
    while count < max_iterations:
        previous = comp.clone()
        comp[img == 1] = F.max_pool3d(comp, kernel_size=3, stride=1, padding=1)[img == 1]
        if torch.equal(comp, previous):
            break
        else:
            count += 1
    for index, elt in enumerate(torch.unique(comp)):
        comp [comp == elt.item()] = index + 1
    return comp

D = W = H = 48
device = 'cuda'
img = torch.randn(1, D, W, H).to(device)

# created Connected components for test
for _ in range(5):
    img = F.avg_pool3d(img, kernel_size=3, stride=1, padding=1)

# Thresholding
threshold = 0.1
img[img > threshold] = 1.0
img[img <= threshold] = 0.0

# Label Connected components
comp = getCC(img)
print(f"Number of CC is: {(torch.numel(torch.unique(comp)) - 1)}")
