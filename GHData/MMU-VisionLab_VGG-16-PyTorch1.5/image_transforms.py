'''
Contains Classes and Methods to be used for random image augmentations during neural network trainings. TorchVision has built-in transformation
methods and classes already but I'm writing these classes manually so that I have more control over the methods. Also TorchVision transformation methods requires the input to be a PIL image. I prefer them to be in NumPy array. The Probability list in the __init__ methods contains a list of 1's and 0's. If the given probability is 0.5 then there will be equal amount of 1's and 0's in the list. The idea here is to randomly choose an element and if it's 1, continue with the augmentation else no augmentation.
'''
import random
from skimage import transform, util
from skimage.transform import rotate, AffineTransform
import torch
import numpy as np


class RandomRotate:
    '''
    Randomly rotates the input image to the given angle range.
    '''

    @staticmethod
    def get_random_rotation(angle_range):
        '''
        Generate a random number between the given range for rotation angle.
        '''
        angle = None
        #check if the angle_range is either one of the listed types.
        assert isinstance(angle_range, (int, list, tuple)), "angle_range must be either tuple, list or integer!"

        #generate a random number between the given range.
        if isinstance(angle_range, int): #if the angle_range is an integer.
            angle = random.randrange(0, angle_range) if angle_range > 0 else random.randrange(angle_range, 0)
        else:
            angle = random.randrange(angle_range[0], angle_range[1]) if angle_range[0] < angle_range[1] else random.randrange(angle_range[1], angle_range[0])

        return angle

    def __init__(self, angle_range, prob=0.5):
        '''
        Intialize parameters. Rotation angle cannot be initialized here as it'll be only initialized once for every batch thus making all the images in a batch to be rotated at the same angle.
        '''
        assert isinstance(angle_range, (int, list, tuple)), "angle_range must be either tuple, list or float!"
        assert 0 <= prob <= 1, "The probability has to be a number from 0 to 1."
        self.angle_range = angle_range
        prob_percentage = int(prob*100)
        self.prob_list = [1]*prob_percentage + [0]*(100-prob_percentage) #list of 1's and 0's. e.g. [1,1,...,0,0,0]

    def __call__(self, sample):
        '''
        Rotates the input image using the generated rotation angle if the probability is 1.
        '''

        if random.choice(self.prob_list) == 0:
            return sample

        sample['image'] = rotate(sample['image'], angle=self.get_random_rotation(self.angle_range)) #rotates the image at the generated angle.

        return sample


class RandomShear:
    '''
    Randomly shears a cv2 image or returns back the same original image based on the given probability. The shearing value will also be randomly generated based on the given range or integer.
    '''

    @staticmethod
    def get_random_value(shear_range):
        '''
        Generates a random value for shearing.
        '''
         #check if the shear_range is either one of the listed types.
        assert isinstance(shear_range, (float, list, tuple)), "shear_range must be either tuple, list or float!"

        shear_value = 0
        if isinstance(shear_range, (float)):
            shear_value = round(random.uniform(0, shear_range), 1) if shear_range > 0 else round(random.uniform(shear_range, 0), 1)
        else:
            shear_value = round(random.uniform(shear_range[0], shear_range[1]), 1) if shear_range[0] < shear_range[1] else round(random.uniform(shear_range[1], shear_range[0]), 1)

        return shear_value


    def __init__(self, shear_range, prob=0.5):
        '''
        Parameter initilization.
        '''
        assert 0 <= prob <= 1, "The probability has to be a number from 0 to 1."
        self.shear_range = shear_range
        prob_percentage = int(prob*100)
        self.prob_list = [1]*prob_percentage + [0]*(100-prob_percentage) #list of 1's and 0's. e.g. [1,1,...,0,0,0]

    def __call__(self, sample):
        '''
        Returns back the original image if the picked element from the list is 0. Else shear the image based on the generated value and return it.
        '''

        if random.choice(self.prob_list) == 0:
            return sample

        image = sample['image']

        shear_obj = AffineTransform(shear=self.get_random_value(self.shear_range))

        sample['image'] = transform.warp(image, shear_obj, order=1, preserve_range=True, mode='wrap')

        return sample

class RandomHorizontalFlip:
    '''
    Returns the original given input image back or returns the flipped image based on the given probability.
    '''
    def __init__(self, prob=0.5):
        '''
        Initiliaze parameters.
        '''
        assert 0 <= prob <= 1, "The probability has to be a number from 0 to 1."
        prob_percentage = int(prob*100)
        self.prob_list = [1]*prob_percentage + [0]*(100-prob_percentage) #list of 1's and 0's. e.g. [1,1,...,0,0,0]

    def __call__(self, sample):
        '''
        Returns back the original image if the picked element from the list is 0. Else, flip the image horizontally and return it.
        '''
        if random.choice(self.prob_list) == 0:
            return sample

        image = sample['image']

        sample['image'] = image[:, ::-1, :] #reverses the columns.

        return sample


class RandomVerticalFlip:
    '''
    Returns the original given input image back or returns the flipped image based on the given probability.
    '''
    def __init__(self, prob=0.5):
        '''
        Initiliaze parameters.
        '''
        assert 0 <= prob <= 1, "The probability has to be a number from 0 to 1."
        prob_percentage = int(prob*100)
        self.prob_list = [1]*prob_percentage + [0]*(100-prob_percentage) #list of 1's and 0's. e.g. [1,1,...,0,0,0]

    def __call__(self, sample):
        '''
        Returns back the original image if the picked element from the list is 0. Else, flip the image vertically and return it.
        '''
        if random.choice(self.prob_list) == 0:
            return sample

        image = sample['image']

        sample['image'] = image[::-1, :, :] #reverses the rows.

        return sample


class RandomNoise:
    '''
    Randomly adds noise on a cv2 image.
    '''

    def __init__(self, mode, prob=0.5):
        '''
        Initialize parameters.
        '''
        assert 0 <= prob <= 1, "The probability has to be a number from 0 to 1."
        assert isinstance(mode, (str, list)), "Noise mode has to either be a string or a list!"

        #creates a list of 1's and 0's. There will be exactly "probs" num of 1's and "100 - probs" num of 0's.
        prob_percentage = int(prob*100)
        self.prob_list = [1]*prob_percentage + [0]*(100-prob_percentage) #list of 1's and 0's. e.g. [1,1,...,0,0,0]
        self.mode = mode #mode of noise

    def __call__(self, sample):
        '''
        Adds noise to the image based on the selected mode.
        '''

        if random.choice(self.prob_list) == 0: #no noise if selected element is 0.
            return sample

        if isinstance(self.mode, str):
            sample['image'] = util.random_noise(sample['image'], mode=self.mode, clip=True)
        else:
            sample['image'] = util.random_noise(sample['image'], mode=random.choice(self.mode), clip=True) #randomly choose one of the mode from the list.

        return sample

class ToTensor:
    '''
    Convert the numpy image to a Tensor.
    '''

    def __init__(self, mode='training'):
        '''
        Initialize parameters.
        '''
        self.mode = mode


    def __call__(self, sample):
        '''
        Conversion to Tensor. If the conversion is for training data, labels will be included as well. Else, only the image will be returned as a tensor.
        '''

        sample['image'] = sample['image'].transpose((2, 0, 1)) #pytorch requires the channel to be in the 1st dimension of the tensor.

        if self.mode == 'training':

            return {'image': torch.from_numpy(sample['image'].copy()).type(torch.FloatTensor),
                    'label': torch.from_numpy(np.asarray(sample['label'], dtype='int32')).type(torch.LongTensor)}

        return {'image': torch.from_numpy(sample['image'].copy())}
