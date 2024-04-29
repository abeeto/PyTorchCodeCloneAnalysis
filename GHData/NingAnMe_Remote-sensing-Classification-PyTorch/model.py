# -*- coding: utf-8 -*
import numpy
import os
import torch
from torch.utils.data import DataLoader


from flyai.model.base import Base


from path import MODEL_PATH
from data import ClassifierDatasetTest
from processor import Processor


TORCH_MODEL_NAME = "model.pkl"


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.check_point = None
        self.net = None
        if os.path.exists(MODEL_PATH):
            self.net = torch.load(MODEL_PATH).to(device)

    def predict(self, **data):
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data).to(device)
        outputs = self.net(x_data)
        prediction = outputs.cpu().data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        labels = []
        dataset_test = ClassifierDatasetTest(datas)
        test_loader = DataLoader(dataset_test, batch_size=1)
        p = Processor()
        with torch.no_grad():
            self.net.eval()
            for x in test_loader:
                outputs = self.net(x.to(device))
                _, pred = torch.max(outputs, 1)
                labels.append(p.output_y(pred[0]))
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))
