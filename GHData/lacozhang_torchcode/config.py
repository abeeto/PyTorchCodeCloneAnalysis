import os
import sys
from utils import get_logger
import re
import datetime


class Config():
    def __init__(self, pipeline):
        self.pipeline = pipeline

        # general config
        self.model_name = re.findall("model_name=([^\s\t]+)", self.pipeline)[0]
        self.model_floder = re.findall(
            "output_path=([^\s\t]+)", self.pipeline)[0] + self.model_name + '/'
        self.output_path = self.model_floder  + "tf-model/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.model_floder + "log.txt"

        # directory for training outputs
        if not os.path.exists(self.model_floder):
            os.makedirs(self.model_floder)

        # create instance of logger
        self.logger = get_logger(self.log_path)

        # embeddings
        self.dim = int(re.findall("dim=([^\s\t]+)", self.pipeline)[0])
        self.dim_char = int(
            re.findall("dim_char=([^\s\t]+)", self.pipeline)[0])
        self.glove_filename = os.path.join(
            os.path.split(sys.path[0])[0],
            "glove.6B/glove.6B.{}d.txt".format(self.dim))

        # trimmed embeddings (created from glove_filename with build_data.py)
        self.trimmed_filename = self.model_floder + "glove.6B.{}d.trimmed.npz".format(
            self.dim)

        # dataset
        self.train_filename = re.findall("train_filename=([^\s\t]+)",
                                         self.pipeline)[0]
        self.test_filename = re.findall("test_filename=([^\s\t]+)",
                                        self.pipeline)[0]
        self.dev_filename = self.test_filename if re.findall(
            "dev_filename=([^\s\t]+)",
            self.pipeline)[0] == "None" else re.findall(
                "dev_filename=([^\s\t]+)", self.pipeline)[0]
        self.domain = re.findall("domain=([^\s\t]+)", self.pipeline)[0]
        self.max_iter = None  # if not None, max number of examples

        # vocab (created from dataset with build_data.py)
        self.words_filename = self.model_floder + "words.txt"
        self.labels_filename = self.model_floder + "labels.txt"
        self.chars_filename = self.model_floder + "chars.txt"
        self.infer_filename = self.model_floder + "test.infer.%s.txt" % datetime.datetime.now(
        ).isoformat()[:19].replace(":", ".")
        # DEFAULT = "nonretailrelated"
        self.DEFAULT = re.findall("DEFAULT=([^\s\t]+)", self.pipeline)[0]

        self.train_embeddings = False
        self.nepochs = int(re.findall("nepochs=([^\s\t]+)", self.pipeline)[0])
        self.max_model_to_keep = 1
        self.dropout = float(
            re.findall("dropout=([^\s\t]+)", self.pipeline)[0])
        self.batch_size = int(
            re.findall("batch_size=([^\s\t]+)", self.pipeline)[0])
        self.LR_method = re.findall("LR_method=([^\s\t]+)", self.pipeline)[0]
        self.LR = float(re.findall("LR=([^\s\t]+)", self.pipeline)[0])
        self.LR_decay = float(
            re.findall("LR_decay=([^\s\t]+)", self.pipeline)[0])
        self.clip = float(re.findall(
            "clip=([^\s\t]+)", self.pipeline)[0])  # if negative, no clipping
        self.nepoch_no_imprv = int(
            re.findall("nepoch_no_imprv=([^\s\t]+)", self.pipeline)[0])
        self.reload = re.findall("reload=([^\s\t]+)",
                                 self.pipeline)[0] == 'True'

        # model hyperparameters
        self.hidden_size = int(
            re.findall("hidden_size=([^\s\t]+)", self.pipeline)[0])
        self.char_hidden_size = int(
            re.findall("char_hidden_size=([^\s\t]+)", self.pipeline)[0])

        # if both chars and crf, only 1.6x slower on GPU
        self.crf = re.findall(
            "crf=([^\s\t]+)", self.pipeline
        )[0] == 'True'  # if crf, training is 1.7x slower on CPU
        self.chars = re.findall(
            "chars=([^\s\t]+)", self.pipeline
        )[0] == 'True'  # if char embedding, training is 3.5x slower on CPU

        # build dev data
        self.dev_ratio = 0.4
        self.build_dev_from_trainset = False
        self.build_dev_from_testset = False