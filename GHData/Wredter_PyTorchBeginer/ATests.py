from Models.RetinaNet.Utility import retinabox300
from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.DefaultsBox import *
from Models.SSD.Utility import show_areas
from Models.Utility.Utility import point_form, box_form
from Models.RetinaNet.RetinaNet import RetinaNet
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = torch.gather(t, 0, torch.tensor([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]))
print(t2)
"""
"""
box = dboxes300()
db = box(order="ltrb")
db = box_form(db)
db = db.view(-1, 37, 4)
image = torch.zeros((3, 300, 300), dtype=torch.long)
target = torch.zeros(1, 4)
shape = db.shape[0]
for i in range(db.shape[0]):
    print(f"Test {i} : {db[i]}")
    show_areas(image, target, db[i], None, f"Test {i}")
print("boxes")
"""
"""
model = RetinaNet(num_classes=1).to(device)
img = torch.zeros((1, 3, 608, 608), dtype=torch.float, device=device)
out = model(img)
print(out)
"""
x = torch.tensor([[1, 1, 1, 1], [2, 1, 2, 1]], dtype=torch.float32)
y = torch.tensor([[3, 1, 2, 4], [5, 2, 1, 8]], dtype=torch.float32)
loss = torch.nn.MSELoss()
l1_pom = []
for i in range(2):
    l1 = loss(x[i], y[i])
    l1_pom.append(l1)
l1 = 0
for l in l1_pom:
    l1 += l
l_pom = []
l2 = 0
for i in range(4):
    l = loss(x[:, i], y[:, i])
    l_pom.append(l)
for l in l_pom:
    l2 += l
print(l1)
print(l2)