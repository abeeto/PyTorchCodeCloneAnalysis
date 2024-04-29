import torchvision.transforms as T
import torch
from PIL import Image
import numpy as np
import cv2 as cv
from datetime import datetime
import os
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as nd


import numbers
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU isn't available, CPU is being used.")


class Utils:
    def __init__(self, model_name: str):
        self.mean = np.array([])
        self.stdv = np.array([])
        self.img_size = 0
        self.model_name = model_name

        # Map desired init functions for each model, one for now
        model_mapping = {"vgg19": self.vgg19_param_init}
        # Init mean, standard deviation and desired image size for resizing
        model_mapping[model_name]()

        self.normalize = T.Normalize(mean=self.mean, std=self.stdv)
        self.resize = T.Resize(size=self.img_size, interpolation=2)

    def vgg19_param_init(self):
        """
        Initializes mean, standard deviaton and desired image size for resizing when using VGG19 model.
        """

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stdv = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = 720

    def load_img(self, path: str = None, resize: bool = True):
        """
        Loading image function.

        :param path: Optional path of the image, default value is None and the dream will be made from noise img.
        :param resize: Resize image to 600 width or keep the original shape.
        :return: PIL resized image.
        """
        if path is None:
            shape = (self.img_size, self.img_size, 3)
            img = Image.fromarray(np.random.uniform(low=0, high=255, size=shape).astype(np.uint8), 'RGB')
        else:
            img = Image.open(path)
            if resize:
                img = self.resize(img)

        return img

    def display_img(self, img_list: list, titles: list, save: bool = False):
        """
        Displays and saves dream images in Output folder if specified.

        :param img_list: List of images to display, first one should always be original image and the rest dream images.
        :param titles: Layers used for creating dream images.
        :param save: Optional argument specifying if dream images should be saved. Default behaviour is not saving them.
        :return:
        """

        print("\n Press ESC to exit.")

        cv.namedWindow("DeepDream", cv.WINDOW_NORMAL)
        cv.resizeWindow("DeepDream", img_list[0].size[0], img_list[0].size[1])

        titles = ["original"] + titles
        i = 0
        n = len(img_list)


        if save:
            path = self.make_save_dir()

        while True:
            if i != 0:
                # We need to denormalize dream image before displaying it
                cv_img = cv.cvtColor(self.denormalize(img_list[i]), cv.COLOR_RGB2BGR)
                if save:
                    filename = path + "/" + self.model_name + "_" + titles[i] + ".jpg"
                    cv.imwrite(filename, cv_img * 255)  # mul by 255 because our img is in range [0,1]
            else:
                cv_img = cv.cvtColor(np.array(img_list[0]), cv.COLOR_RGB2BGR)

            cv.imshow("DeepDream", cv_img)

            k = cv.waitKey(100)

            if k == 100:
                i = (i+1)%n
            if k == 97:
                i = (i-1)%n
            if k == 27:
                break

        cv.destroyAllWindows()

    def make_save_dir(self):
        """
        Makes a dir in Output folder with unique name corresponding to the current date and time.

        :return: path of the created folder
        """

        today = datetime.now()
        path = "./Output/" + today.strftime('%H_%M_%S_%d_%m_%Y')
        os.mkdir(path)

        return path

    def save_img(self, img, layer, run, path):
        """
        Saves the image with info about its layer and run in which it was generated.

        :param img: Image to be saved.
        :param layer: Which layer was used to produce the image.
        :param run: In which run was the image generated
        :param path: Path of the directory where the image should be stored.
        :return:
        """

        if run != 0:
            cv_img = cv.cvtColor(self.denormalize(img), cv.COLOR_RGB2BGR)
        else:
            cv_img = np.array(img)
            #cv_img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

        filename = path + "/" + self.model_name + "_" + layer + "_" + str(run) + ".jpg"
        cv.imwrite(filename, cv_img * 255)  # mul by 255 because our img is in range [0,1]

    def clip(self, tensor: torch.Tensor):
        """
        Clips the image in desired [min,max] pixel bounds.

        :param tensor: Image to be clipped.
        :return: Clipped tensor.
        """

        '''
        for channel in range(tensor.shape[1]):
            ch_m, ch_s = self.mean[channel], self.stdv[channel]
            tensor[0, channel] = torch.clamp(tensor[0, channel], -ch_m / ch_s, (1 - ch_m) / ch_s)
        '''

        LOWER_IMAGE_BOUND = torch.tensor((-self.mean / self.stdv).reshape(1, -1, 1, 1)).to('cuda')
        UPPER_IMAGE_BOUND = torch.tensor(((1 - self.mean) / self.stdv).reshape(1, -1, 1, 1)).to('cuda')

        tensor.data = torch.max(torch.min(tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

        return tensor

    def denormalize(self, tensor: torch.Tensor):
        """
        Denormalizes a tensor by multiplying it by stdv and adding mean, and then converts to a PIL image

        :param tensor: Tensor to be denormalized.
        :return: Denormalized tensor.
        """

        tensor = tensor.squeeze(0)  # Tensor format is [batch_sz, channels, w, h] we remove batch_sz which is 1 always
        stdv = self.stdv.reshape((3, 1, 1))
        mean = self.mean.reshape((3, 1, 1))

        tensor = (tensor * stdv) + mean  # Inverse of normalization

        return tensor.numpy().transpose(1, 2, 0)

    def random_circular_spatial_shift(self, tensor, h_shift, w_shift, should_undo=False):
        """
        Shifts the elements of the tensor in the given direction.
        Elements that are shifted beyond the last position are re-introduced at the first position.

        :param tensor: Tensor to be shifted.
        :param h_shift: Number of pixel to shift horizontally.
        :param w_shift: Number of pixel to shift vertically.
        :param should_undo: If True the function will undo the shift.
        :return:
        """
        if should_undo:
            h_shift = -h_shift
            w_shift = -w_shift
        with torch.no_grad():
            rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
            rolled.requires_grad = True
            return rolled

    def gausian_blur(self, grad: torch.tensor, sigma):
        """
        Blurs the gradient.

        :param grad: Grad.
        :param sigma: Parameter for blurring the gradient.
        :return:
        """
        grad = grad.data.cpu()
        grad_1 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad_2 = gaussian_filter(grad, sigma=sigma * 1.0)
        grad_3 = gaussian_filter(grad, sigma=sigma * 2.0)
        return torch.tensor(grad_1 + grad_2 + grad_3, device=device)

    def prepare_new_input_from_output(self, output_img: torch.Tensor, zoom_factor=1.005, width=600, height=800):
        """
        Denormalizes and zooms image from the output tensor to be used again in the dreaming process.
        ( Unnecessary process since we will be converting it again back to tensor and normalizing,
          but requires some time to rewrite other code. )

        :param output_img: Image to be processed.
        :param zoom_factor: Zooming factor.
        :param width: Width before zooming so we can resize it back to that size again.
        :param height: Height before zooming so we can resize it back to that size again.
        :return:
        """

        output_img = self.denormalize(output_img)
        output_img = (output_img * 255).astype(np.uint8)
        output_img = nd.zoom(output_img, zoom_factor)

        new_height = height * zoom_factor
        new_width = width * zoom_factor

        left_limit = int((new_height - height) / 2)
        right_limit = int((new_width - width) / 2)
        output_img = output_img[left_limit:height + left_limit, right_limit:width + right_limit]

        return Image.fromarray(output_img, 'RGB')

# Copied from another DeepDream implementation repo.
class CascadeGaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing seperately for each channel (depthwise convolution)
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers

        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size / 2)  # Used to pad the channels so that after applying the kernel we have same size

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        for sigma in sigmas:
            kernel = 1
            for size_1d, std_1d, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        prepared_kernels = []
        for kernel in kernels:
            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
            kernel = kernel.to('cuda')
            prepared_kernels.append(kernel)

        self.register_buffer('weight1', prepared_kernels[0])
        self.register_buffer('weight2', prepared_kernels[1])
        self.register_buffer('weight3', prepared_kernels[2])
        self.conv = torch.nn.functional.conv2d

    def forward(self, input):
        """
        Apply gaussian filter to input.
        """
        input = torch.nn.functional.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')
        grad1 = self.conv(input, weight=self.weight1, groups=3)
        grad2 = self.conv(input, weight=self.weight2, groups=3)
        grad3 = self.conv(input, weight=self.weight3, groups=3)
        return grad1 + grad2 + grad3
