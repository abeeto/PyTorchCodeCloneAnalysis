# @Author : bamtercelboo
# @Datetime : 2018/8/24 15:27
# @File : test.py
# @Last Modify Time : 2018/8/24 15:27
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  test.py
    FUNCTION : None
"""
import os
import sys
import torch
import json
from DataUtils.utils import *


def load_test_model(model, config):
    """
    :param model:  initial model
    :param config:  config
    :return:  loaded model
    """
    if config.t_model is None:
        test_model_dir = config.save_best_model_dir
        test_model_name = "{}.pt".format(config.model_name)
        test_model_path = os.path.join(test_model_dir, test_model_name)
        print("load default model from {}".format(test_model_path))
    else:
        test_model_path = config.t_model
        print("load user model from {}".format(test_model_path))
    model.load_state_dict(torch.load(test_model_path))
    return model


def load_test_data(train_iter=None, dev_iter=None, test_iter=None, config=None):
    """
    :param train_iter:  train data
    :param dev_iter:  dev data
    :param test_iter:  test data
    :param config:  config
    :return:  data for test
    """
    data, path_source, path_result = None, None, None
    if config.t_data is None:
        print("default[test] for model test.")
        data = test_iter
        sort_path = "_".join([config.test_file, "sort.json"])
        path_source = sort_path if os.path.exists(sort_path) else config.test_file
        path_result = "{}_out.json".format(path_source)
    elif config.t_data == "train":
        print("train data for model test.")
        data = train_iter
        sort_path = "_".join([config.train_file, "sort.json"])
        path_source = sort_path if os.path.exists(sort_path) else config.train_file
        path_result = "{}_out.json".format(path_source)
    elif config.t_data == "dev":
        print("dev data for model test.")
        data = dev_iter
        sort_path = "_".join([config.dev_file, "sort.json"])
        path_source = sort_path if os.path.exists(sort_path) else config.dev_file
        path_result = "{}_out.json".format(path_source)
    elif config.t_data == "test":
        print("test data for model test.")
        data = test_iter
        sort_path = "_".join([config.test_file, "sort.json"])
        path_source = sort_path if os.path.exists(sort_path) else config.test_file
        path_result = "{}_out.json".format(path_source)
    else:
        print("Error value --- t_data = {}, must in [None, 'train', 'dev', 'test'].".format(config.t_data))
        exit()
    return data, path_source, path_result


class T_Inference(object):
    """
        Test Inference
    """
    def __init__(self, model, data, path_source, path_result, alphabet, config):
        """
        :param model:  nn model
        :param data:  infer data
        :param path_source:  source data path
        :param path_result:  result data path
        :param alphabet:  alphabet
        :param config:  config
        """
        print("Initialize T_Inference")
        self.model = model
        self.data = data
        self.path_source = path_source
        self.path_result = path_result
        self.alphabet = alphabet
        self.config = config

    def infer2file(self):
        """
        :return: None
        """
        print("Infer.....")
        self.model.eval()
        predict_label = []
        all_count = len(self.data)
        now_count = 0
        accusation_count = self.config.accu_class_num
        print("accusation count is {}.".format(accusation_count))
        for data in self.data:
            now_count += 1
            sys.stdout.write("\rInfer with batch number {}/{} .".format(now_count, all_count))
            accu, law, prison, e_time, d_time = self.model(data)
            predict = getMaxindex_batch(accu.view(accu.size(0) * accu.size(1), -1))
            predict = self._chunks(predict, accusation_count)
            for i in range(len(predict)):
                elem_index = self._elem_index(predict[i], e=1)
                accu_label = self._id2label(elem_index)
                predict_label.append(accu_label)
        # print(predict_label)
        print("\nInfer Finished.")
        self._write2file(result=predict_label, path_source=self.path_source, path_result=self.path_result)

    def _id2label(self, elem):
        """
        :param elem:
        :return:
        """
        return [self.alphabet.accu_label_alphabet.from_id(i) for i in elem]

    @staticmethod
    def _chunks(L, n):
        """
        :param arr:
        :param n:
        :return:
        """
        return [L[i:i + n] for i in range(0, len(L), n)]

    @staticmethod
    def _elem_index(L, e):
        """
        :param L:  list
        :param e:
        :return:
        """
        return [i for i, v in enumerate(L) if v == e]

    @staticmethod
    def _write2file(result, path_source, path_result):
        """
        :param result:
        :param path_source:
        :param path_result:
        :return:
        """
        print("write result to file {}".format(path_result))
        if os.path.exists(path_source) is False:
            print("source data path[path_source] is not exist.")
        if os.path.exists(path_result):
            os.remove(path_result)
        file_out = open(path_result, encoding="UTF-8", mode="w")

        with open(path_source, encoding="UTF-8") as file:
            id = 0
            for line in file.readlines():
                sys.stdout.write("\rwrite with {}/{} .".format(id+1, len(result)))
                w_line = line
                line_json = json.loads(w_line)
                line_json["meta"]["predict_accu"] = result[id]
                jsObj = json.dumps(line_json, ensure_ascii=False)
                file_out.write(jsObj + "\n")
                id += 1
                if id >= len(result):
                    break

        file_out.close()
        print("\nFinished.")








