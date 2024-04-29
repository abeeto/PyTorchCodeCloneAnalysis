
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from config import HR_SIZE, LR_SIZE
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

def preprocess_train(img):
    """
    img <PIL Image> : Original image
    returns lr <tensor>, hr <tensor> cropped and transformed 
    """
    both_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop([HR_SIZE,HR_SIZE]),
        transforms.RandomVerticalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomInvert()
        
    ])
    hr_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    lr_tfm = transforms.Compose([
        transforms.Resize([LR_SIZE, LR_SIZE], InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    img = both_tfm(img) 

    return lr_tfm(img), hr_tfm(img)


test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

"""returns <PIL Image>"""
inverse_tfm = transforms.Compose([
    transforms.Normalize(mean = [0,0,0], std = [1/s for s in std]),
    transforms.Normalize(mean = [-m for m in mean], std = [1,1,1]),
    transforms.ToPILImage()
])




# %%
