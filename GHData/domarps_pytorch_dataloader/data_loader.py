import numpy as np
import io, os, glob, json
from PIL import Image
import base64
import torch
import torch.utils.data
import urllib.request

class FlickerDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_files):
        self.transform =  None
        self.items = []
        for chunk_file in chunk_files:
            chunk_items = self._read_chunk_file(chunk_file)
            self.items.extend(chunk_items)


    def __len__(self):
        return len(self.items)

    def _read_chunk_file(self, chunk_file):
        try:
            with open(chunk_file, 'r') as f:
                items = list(f)
                return items
        except:
            return []


    def get_PIL_from_url(self, url):
      with urllib.request.urlopen(url) as response:
        image_file = io.BytesIO(response.read())
        im = Image.open(image_file)
        return im

    def preprocess(self, img, size=224):
        img_ = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)
        img_ -= np.array([[[103.939, 116.779, 123.68]]])
        img_ = img_[:,:,::-1]
        return img_.transpose(2,0,1)[np.newaxis,:]

    def _prepare_item(self, item):
        cid, url = item.strip().split(',')
        pil_image = self.get_PIL_from_url(url)
        try:
            image_data = self.preprocess(pil_image)
        except Exception as e:
            image_data = None
        new_item = (cid, image_data)
        return new_item

    def __getitem__(self, index):
        item = self.items[index]
        prepared_item = self._prepare_item(item)
        cid, image_data = prepared_item
        if image_data is None:
            return None
        else:
            return cid, image_data


def filtered_collate_fn(batch):
  # Skip errors. __get_item__ returns None if catches an exception.
  return torch.utils.data.dataloader.default_collate([x for x in batch if x is not None])
