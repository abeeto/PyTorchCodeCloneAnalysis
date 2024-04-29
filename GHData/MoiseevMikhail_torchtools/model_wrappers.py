from abc import ABC, abstractmethod
import dataclasses
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from adam_core.tracing.instrumentation import traced_function
from torchvision import transforms
from modules.document.sara_dataflow import SaraImage, SaraText
import modules.document.constants as const


class JITBasicWrapper(ABC):
    def __init__(
            self,
            weights_dir: str,
            model_file: str,
            device: Union[str, torch.device],
            size: Tuple[int, int] = (256, 256),
            normalize: Optional[tuple] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            threshold: Optional[float] = None,
            **kwargs,
    ):
        """
        :param weights_dir: model directory
        :param model_file: model file name
        :param device: device name (e.g. 'cuda:0', 'cpu')
        :param size: input image size (height, width)
        :param normalize: mean and std stats per channel: ((mean0, mean1, mean2), (std0, std1, std2))
            default are imagenet stats
        :param threshold: use for various postprocessings (label selection in multilabel classification,
            mask filtering in segmentation, etc)
        """
        self.device = str(device)
        self.model = torch.jit.load(
            os.path.join(weights_dir, model_file),
            map_location=self.device,
        )
        self.model.eval()
        if normalize is None:
            self.normalize = None
        else:
            self.normalize = transforms.Normalize(mean=normalize[0], std=normalize[1])
        self.size = size
        self.threshold = threshold
        self._img_sizes = None
        self._input = None
        self.model_output = None
        self.none_indices = []

    def activation(self, model_output):
        return model_output

    def preprocess(self, x):
        return x

    def _remove_nones(self, image_list):
        self.none_indices = [i for i, img in enumerate(image_list) if img is None]
        if self.none_indices:
            image_list = [img for img in image_list if img is not None]
        return image_list

    @traced_function
    def prepare_batch(self, image_list):
        batch = torch.ones((len(image_list), 3, self.size[0], self.size[1]), requires_grad=False).to(self.device)
        for index, img in enumerate(image_list):
            img_tensor = self.img2tensor(img)
            batch[index, :] = img_tensor
        return batch

    def img2tensor(self, image):
        int_method = cv2.INTER_CUBIC
        if image.shape[0] > self.size[0]:
            int_method = cv2.INTER_AREA
        img_tensor = cv2.resize(image, self.size, interpolation=int_method) / 255
        img_tensor = torch.Tensor(img_tensor.astype('float32')).permute(2, 0, 1)
        if self.normalize is not None:
            img_tensor = self.normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    @abstractmethod
    def postprocess(self, model_output):
        raise NotImplementedError

    def wrap_output(self, result):
        return result

    def _insert_nones(self, wrapped_result):
        if self.none_indices is not None:
            self.none_indices.reverse()
            for idx in self.none_indices:
                wrapped_result.insert(idx, None)
        return wrapped_result

    @traced_function(operation_name='jit_predict')
    def predict(self, x):
        image_list = self.preprocess(x)
        image_list = self._remove_nones(image_list)
        if not image_list:
            return self._insert_nones([])
        if not isinstance(image_list, list):
            image_list = [image_list]
        batch = self.prepare_batch(image_list)
        self._img_sizes = [img.shape[:2] for img in image_list]
        with torch.no_grad():
            self.model_output = self.model(batch)
        self.model_output = self.activation(self.model_output)
        self.model_output = self.model_output.data.cpu().numpy()
        result = self.postprocess(self.model_output)
        wrapped_result = self.wrap_output(result)
        wrapped_result = self._insert_nones(wrapped_result)
        return wrapped_result

    def __call__(self, x):
        return self.predict(x)


class JITClassifier(JITBasicWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = kwargs['classes']

    def postprocess(self, model_output):
        assert model_output.shape[1] == len(self.classes), \
            f'number of classes {len(self.classes)} not equal to model output shape[1] {model_output.shape[1]}'
        result = [self.classes[single_img_logits.argmax()] for single_img_logits in model_output]
        return result


class JITMultiLabelClf(JITClassifier):

    def activation(self, model_output):
        return torch.sigmoid(model_output)

    def _get_single_img_labels(self, single_img_proba):
        return [self.classes[idx] for idx, proba in enumerate(single_img_proba) if proba > self.threshold]

    def postprocess(self, activated_output):
        assert activated_output.shape[1] == len(self.classes), \
            f'number of classes {len(self.classes)} not equal to model output shape[1] {activated_output.shape[1]}'
        result = list(map(self._get_single_img_labels, activated_output))
        return result


class JITClfWithProba(JITClassifier):

    def activation(self, model_output):
        return F.softmax(model_output)

    def postprocess(self, activated_output):
        result = {'value': [], 'confidence': []}
        for single_img_proba in activated_output:
            result['value'].append(self.classes[single_img_proba.argmax()])
            result['confidence'].append(single_img_proba.max())
        return result


class JITSegmentation(JITBasicWrapper):

    def postprocess(self, model_output):
        masks = np.argmax(model_output, axis=1).astype('uint8')
        return list(masks)
