"""
    Main Handler Script
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import io
import torch
from PIL import Image
from requests import request
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class Handler(BaseHandler):
    """ Custom handler for pytorch serve """

    def __init__(self, *args, **kwargs):
        super(Handler, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self.args = args
        self.kwargs = kwargs

    def preprocess_one_image(self, req: request):
        """
        Preprocess one single image
        :param req: request
        :return: preprocessed image
        """
        image = req.get("data")
        if image is None:
            image = req.get("body")
        image - Image.open(io.BytesIO(image))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a tensor
        :param requests: request
        :return: batch of processed images
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, data, *args, **kwargs):
        """
        Performing model inference
        :param data: Given data from preprocess method
        :param args: Optional arguments
        :param kwargs: Optional keyword arguments
        :return: predicted label for each image
        """
        outs = self.model.forward(data)
        probs = F.softmax(outs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def postprocess(self, data):
        """
        Postprocess the output
        :param data: Given data from inference method
        :return: Human readable label
        """
        res = []
        preds = data.cpu().tolist()
        for pred in preds:
            label = self.mapping[str(pred)][1]
            res.append({"label": label, "index": pred})
        return res
