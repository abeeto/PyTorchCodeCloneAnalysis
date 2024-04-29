import torchvision.transforms as transforms

def make_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразовать в тензор и нормализовать до [0, 1]
    ])
    return transform