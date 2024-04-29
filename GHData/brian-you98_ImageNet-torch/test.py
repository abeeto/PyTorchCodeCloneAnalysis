import torch
import cv2

a = cv2.imread("E:\PyCoding\other\zhanchen\Contrast\DataSources\\train\hazy\\train (1).png")
b = cv2.imread("E:\PyCoding\other\zhanchen\Contrast\DataSources\\train\clean\\train (1).png")


a_f = torch.fft.fft2(torch.Tensor(a))
b_f = torch.fft.fft2(torch.Tensor(b))
a_real = a_f.real
a_imag = a_f.imag
b_real = b_f.real
b_imag = b_f.imag

shape_scale = a.shape[0] * a.shape[1]
a_amp = torch.sqrt(torch.square(a_real) + torch.square(a_imag)) / shape_scale
a_pha = torch.atan2(a_imag, a_real) / torch.Tensor([3.1415926])
b_amp = torch.sqrt(torch.square(b_real) + torch.square(b_imag)) / shape_scale
b_pha = torch.atan2(b_imag, b_real) / torch.Tensor([3.1415926])

d = torch.sqrt(torch.square(a_amp - b_amp) + torch.square(a_pha - b_pha))
