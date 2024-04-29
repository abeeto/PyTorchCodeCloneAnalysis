import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from model import MNISTClassifier
from utils import get_device


class MNISTHandler(BaseHandler):
    def __init__(self):
        super(MNISTHandler, self).__init__()
        self.device = get_device()
        self.model = MNISTClassifier()
        self.model.load_state_dict(torch.load('mnist_model.pth', map_location=self.device))
        self.model.eval()
        print(os.listdir('.'))

 
    def preprocess(self, requests):
        """
        Preprocess batch data from the requests and return batched tensor.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((28, 28)),
        ])
        
        images = []
        for request in requests:
            img_bytes = request.get('body')
            if img_bytes is None:
                continue
                
            image = Image.open(io.BytesIO(img_bytes)).convert('L')
            image = transform(image)
            image = image.unsqueeze(dim=0)
            images.append(image)

        images = torch.cat(images)

        return images


    def inference(self, x):
        outputs = self.model.forward(x)
        preds = F.softmax(outputs, dim=-1) # (batch_size, 10)

        return preds

    
    def postprocess(self, preds):
        """
        결과를 list로 반환해야함.
        ex) List of Dictionary
        """
        probs = preds.cpu().tolist()

        results = []
        for prob in probs:
            result = {}
            for label, p in enumerate(prob):
                result[label] = p
            results.append(result)

        return results
