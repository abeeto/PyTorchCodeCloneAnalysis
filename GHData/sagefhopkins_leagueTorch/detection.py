from detecto import core, utils, visualize
from torchvision import transforms
import cv2


def trainDataset():
    dataset = core.Dataset('images\\Ahri')
    model = core.Model(['Aatrox', "Ahri", "Akali"])
    model.fit(dataset, verbose=True)
    model.save('model.pth')


trainDataset()
