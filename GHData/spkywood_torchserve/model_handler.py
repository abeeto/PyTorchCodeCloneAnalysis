from unicodedata import name
import io
import os
import time
import torch
import base64
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ModelHandler(object):
    
    def __init__(self):
       self.model = None
       self.device = None
       self.initialized = False
       self.context = None
       self.manifest = None
       self.map_location = None
       self.explain = False

    """
        load model
    """
    def initialize(self, context):
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        model_name = self.manifest["model"]["serializedFile"]
        # model_name = os.path.splitext(serialized_file)[0]
        model_path = os.path.join(model_dir, model_name)

        if not os.path.isfile(model_path):
            raise RuntimeError("Missing the model.pt file")

        # pytorch model load
        # 模型保存的是网络和权重 
        self.model = torch.load(model_name, map_location=self.device)
        logger.info("Loading eager model")
        self.model.eval().to(self.device)

        logger.info('Model file %s loaded successfully', model_path)
        self.initialized = True

    # Http 请求数据处理
    def preprocess(self, data):
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return images

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            output = self.model(data_preprocess)
            output = self.postprocess(output)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output

    def _is_explain(self):
        return False

    def postprocess(self, data):
        result = []
        """
            model result 处理
            分类，检测，语义分割...
        """
        return result

    def interface(self, model_input):
        pass
