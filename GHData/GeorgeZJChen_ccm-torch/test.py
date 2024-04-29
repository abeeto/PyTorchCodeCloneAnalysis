from __future__ import print_function, division
import torch
import numpy as np
from PIL import Image, ImageOps
import os
import random
import sys
import time
import pickle

from functions import *
from data import *
from model import *

def get_test_names(part='B'):
    if not (os.path.exists('./test_dict.pkl') and os.path.exists('./strict_test_names.pkl')):
        # tf.reset_default_graph()
        test_dict = {}
        strict_test_names = preprocess_data(
            names=load_data_names(train=False, part=part),
            data_path='./datasets/shanghaitech/'+part+'/test/',
            test=True,
            test_dict=test_dict,
            input_size=[384,512]
        )
        random.shuffle(strict_test_names)
        print()
        print(len(strict_test_names), 'of data')
        with open('strict_test_names.pkl', 'wb') as f:
            pickle.dump(strict_test_names, f)
        with open('test_dict.pkl', 'wb') as f:
            pickle.dump(test_dict, f)
    else:
        strict_test_names = pickle.load(open('./strict_test_names.pkl', 'rb'))
        test_dict = pickle.load(open('./test_dict.pkl', 'rb'))
    return strict_test_names, test_dict

def get_data_by_name(name):

  img = np.asarray(Image.open(name+'.jpg'))
  target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
  target15 = np.reshape(np.sum(target14), [1,1,1])

  targets = [[target15], [target14], [target13], [target12], [target11], [target10]]
  return np.array(normalize([img])), [np.array(t) for t in targets]


def full_test(net, part='B'):
  strict_test_names, test_dict = get_test_names(part)

  print(">>> Test begins", end='.')

  for key in test_dict:
    test_dict[key]['predict'] = np.array([0.0]*6)
    test_dict[key]['truth'] = 0

  step = 0
  for test_name_strict in strict_test_names:

    test_inputs, test_targets = get_data_by_name(test_name_strict)
    test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets

    test_D, test_loss = net.test(test_inputs, test_targets)

    out15, out14, out13, out12, out11, out10 = [ to_np(tensor) for tensor in test_D ]
    data_name = test_dict['names_to_name'][test_name_strict]
    test_dict[data_name]['predict'] += np.array([np.sum(out15),np.sum(out14),np.sum(out13),np.sum(out12),np.sum(out11),np.sum(out10)])
    test_dict[data_name]['truth'] += np.sum(test_t15)

    step += 1

    if step % (len(strict_test_names)//15) == 0:
        print('.', end='')
        sys.stdout.flush()
  print()

  results = []
  for key in test_dict:
    if key != 'names_to_name':
      _data = test_dict[key]
      results.append(np.abs(_data['predict']-_data['truth']))

  results = np.mean(results, axis=0)
  return results

def test():
    # TODO:
    pass

if __name__ == "__main__":
    test()
