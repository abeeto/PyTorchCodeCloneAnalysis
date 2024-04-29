import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
import cv2
import os
import base64
import numpy as np
import json
import time
from minio import Minio
from model import ExpandNet
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
from util import (
    process_path,
    split_path,
    map_range,
    str2bool,
    cv2torch,
    torch2cv,
    resize,
    tone_map,
    create_tmo_param_from_args,
)
class RGBINOUTHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.DefaultFullPath = "/tmp/foo.jpg"
        self.DefaultProcessedFullPath = "/tmp/processed.jpg"
        self.opt = {
            "gamma": 1.0,
            "height": 540,
            "ldr": "",
            "ldr_extensions": ['.jpg', '.jpeg', '.tiff', '.bmp', '.png'],
            "out": "",
            "patch_size": 256,
            "resize": False,
            "stops": 0.0,
            "tag": None,
            "tone_map": "reinhard",
            "use_exr": False,
            "use_gpu": True,
            "use_weights": "weights.pth",
            "video": False,
            "width": 960,
        }

        self.net = self.load_pretrained(self.opt)

    def load_pretrained(self,opt):
        net = ExpandNet()
        net.load_state_dict(
            torch.load(opt["use_weights"], map_location=lambda s, l: s)
        )
        net.eval()
        return net
    
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def DownloadFromS3(self, data, client):
        bucketname = data[0]["body"]["bucket"]
        filename = data[0]["body"]["file"]
        logging.debug("Bucket %s" % data[0]["body"]["bucket"])
        logging.debug("file %s" % data[0]["body"]["file"])
        try:
            client.fget_object(bucketname,filename,self.DefaultFullPath)
        except:
            raise PredictionException("Some Prediction Error", 501)
            return False
        
        return bucketname, filename
        

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        
        loaded = cv2.imread(self.DefaultFullPath, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        loaded = loaded.astype('float32')
        loaded = map_range(loaded)
        return loaded






    def inference(self, model_input):
        
        t_input = cv2torch(model_input) # TODO: add support for batch of images
        if self.opt["use_gpu"]:
            self.net.cuda()
            t_input = t_input.cuda()
        prediction = map_range(
            torch2cv(self.net.predict(t_input, 256).cpu()), 0, 1
        )

        return prediction

    def postprocess(self, inference_output ):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """        
        postprocess_output = inference_output
        if self.opt["tone_map"] is not None:
                postprocess_output = tone_map(
                    postprocess_output, self.opt["tone_map"], **create_tmo_param_from_args(self.opt)
                )
        else:
                raise PredictionException("Some Prediction Error", 501)
        cv2.imwrite(self.DefaultProcessedFullPath, (postprocess_output * 255).astype(int))
      
        return postprocess_output

    def UploadToS3(self,client, bucketname, filename):
        client.fput_object(bucketname, "processed.jpg", self.DefaultProcessedFullPath,)

    def HandleConfig(self, data):
        
        if "config" in data[0]["body"] :
            config = data[0]["body"]["config"]
            print("Found Config: %s", config)
            for i in config:
                print("Key: %s" % i)
                if i in self.opt:
                    print("Updating Key %s with Value: %s" % (i, config[i]))
                    self.opt[i] = config[i]
                    print("sucessfull updated")

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        start = time.time()
        result = {}
        res = 0
        try:
            s3url = os.environ.get("S3Url", "storage.XXXXXX.com")
            s3accesskey = os.environ.get("S3AccessKey", "BXXXXXOrOJ0m0FPGt")
            s3secretkey = os.environ.get("S3SecretKey", "RSSEQMLXXXXXXX0CAP26sJAzL4uW")
            client = Minio(
                s3url,
                access_key=s3accesskey, # hier natürliuch os.env
                secret_key=s3secretkey, # hier auch
                secure=True
            )
            self.HandleConfig(data)
            bucketname, filename = self.DownloadFromS3(data, client)
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)  
            res =  self.postprocess(model_output)
            self.UploadToS3(client, bucketname, filename)
            end = time.time()
            result["executionTime"] = (end-start)
            result["bucket"] = bucketname
            result["filename"] = filename
        except:    
            raise PredictionException("Some Prediction Error", 501)
        
        result["Success"] = True
        returnset = [
            result            
        ]
        return returnset

        