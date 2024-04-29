import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms

from serve.ts.torch_handler.base_handler import BaseHandler
import argparse
import numpy as np

from model import R2GenModel
from PIL import Image


class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])
        
        self.model = R2GenModel()

        device = torch.device('cpu')

        self.model.load_state_dict(torch.load('model_iu_xray.pth',map_location=torch.device('cpu'))['state_dict'])

        self.model = self.model.to(device)
        

    # def preprocess_one_image(self, req):
    #     """
    #     Process one single image.
    #     """
    #     # get image from the request
    #     image = req.get("data")
    #     if image is None:
    #         image = req.get("body")       
    #      # create a stream from the encoded image
    #     image = Image.open(io.BytesIO(image))
    #     image = self.transform(image)
    #     # add batch dim
    #     image = image.unsqueeze(0)
    #     return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        # images = [self.preprocess_one_image(req) for req in requests]
        # images = torch.cat(images)    
        # return images

        image = requests.get("data")  #data-> [1.png,2.png]
        if image is None:
            image = requests.get("body")  
        # image1 = Image.open(image1).convert('RGB')
        # image2 = Image.open(image2]).convert('RGB')

        image1 = self.transform(image[0])
        image2 = self.transform(image[1])

        image = torch.stack((image1, image2))
        image = torch.unsqueeze(image, 0)
        image.shape
        device = torch.device('cpu')
        image = image.to(device)

        return image

    
    def inference(self, image):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        # outs = self.model.forward(x)
        # probs = F.softmax(outs, dim=1) 
        # preds = torch.argmax(probs, dim=1)
        # return preds
        prediction = self.model(image, mode='sample')

        return prediction



    def postprocess(self, prediction):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        # res = []
        # # pres has size [BATCH_SIZE, 1]
        # # convert it to list
        # preds = preds.cpu().tolist()
        # for pred in preds:
        #     label = self.mapping[str(pred)][1]
        #     res.append({'label' : label, 'index': pred })
        # return res

        return self.model.tokenizer.decode_batch(prediction.cpu().numpy())
