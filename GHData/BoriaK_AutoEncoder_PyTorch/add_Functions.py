import torch
from PIL import Image
import io
# import torchac
import torchvision.transforms as transforms

datasetFolder = r'../dataSet/DataSet1'


def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def JPEGCompression(img_ten):
    # takes a Tensor, and returnes a JPEG converted PIL Image, converted to tensor
    # ce = range(0, 10)   #bits per pixel
    image = transforms.functional.to_pil_image(img_ten)
    outputIoStream = io.BytesIO()
    image.save(outputIoStream, "JPEG", quality=75, optimice=True)
    outputIoStream.seek(0)
    # bpp = Image.open(outputIoStream).bits
    img_ten_jpeg = transforms.ToTensor()(Image.open(outputIoStream)).unsqueeze_(0)
    return img_ten_jpeg


def JPEG2000Compression(img_ten, bpp):
    # takes a Tensor, and returnes a JPEG converted PIL Image, converted to tensor
    # ce = range(0, 10)   #bits per pixel
    image = transforms.functional.to_pil_image(img_ten)
    outputIoStream = io.BytesIO()
    image.save(outputIoStream, "JPEG2000", quality_mode='rates', quality_layers=[bpp], optimice=True)
    outputIoStream.seek(0)
    # bpp = Image.open(outputIoStream).bits
    img_ten_jpeg2K = transforms.ToTensor()(Image.open(outputIoStream)).unsqueeze_(0)
    return img_ten_jpeg2K
