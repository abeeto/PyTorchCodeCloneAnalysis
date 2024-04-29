import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import PIL
from PIL import Image


#Transform class for pairwise data augmentation, flip, random crop, random rotation, ColorJitter (possibly useful for denoising), you can add your transformation also.
class PAIR_TRANFORMATIONS(object, h_flip=False, v_flip=False, Random_crop=False, Random_rotation_degree=0, ColorJitter=[0,0,0,0]):

    def __call__(self, input, target):


        #Horizontal flip
        if h_flip:
            flip = random.random() < 0.5
            if flip:
                input=input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)


        #Vertical flip
        if v_flip:
            flip = random.random() < 0.5
            if flip:
                input = input.transpose(Image.FLIP_TOP_BOTTOM)
                target = target.transpose(Image.FLIP_TOP_BOTTOM)


        #RandomCrop
        if Random_crop:
            i, j, h, w = transforms.RandomCrop.get_params(input, output_size=(224, 224))
            input = TF.crop(input, i, j, h, w)
            target = TF.crop(target, i, j, h, w)


        #RandomRotation
        i = transforms.RandomRotation.get_params(angle)
        input = TF.rotate(input, i)
        target = TF.rotate(target, i)



        #ColorJitter
        b,c,s,h=transforms.ColorJitter.get_params(ColorJitter[0],ColorJitter[1],ColorJitter[2],ColorJitter[3])

        input=TF.adjust_brightness(input,b)
        input=TF.adjust_contrast(input,c)
        input = TF.adjust_saturation(input, s)
        input = TF.adjust_hue(input, h)

        target=TF.adjust_brightness(target,b)
        target=TF.adjust_contrast(target,c)
        target = TF.adjust_saturation(target, s)
        target = TF.adjust_hue(target, h)


        return input, target

