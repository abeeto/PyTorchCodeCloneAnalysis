from data_utils import load_data
from train import apply_cuda
from datetime import datetime
import torch
import numpy as np

def export_test_data_to_numpy(images, labels, data_folder, n_max=1000):
  import os
  data_path = os.path.join(data_folder, 'test_data.npz')
  np.savez('data/test_data.npz', images=images[:n_max], labels=labels[:n_max])

def test(model_name, model_ckpt, dataset_name, data_folder):
  # model definition
  if model_name == 'lenet':
    from model.lenet import LeNet
    model = LeNet()
  else:
    from model.modelzoo import create_model
    model, input_size = create_model(model_name, n_classes=120)
  model = apply_cuda(model)

  # load weights
  ckpt = torch.load(model_ckpt)
  model.load_state_dict(ckpt['state_dict'])
  
  # data source
  batch_size = 200
  if dataset_name == 'mnist':
    test_loader = load_data('test', batch_size, data_folder, dataset_name)
  else:
    test_loader = load_data('test', batch_size, data_folder, dataset_name, input_size)
  n_batches_test = len(test_loader)
  
  print('==== test phase ====')
  avg_acc = float(0)
  model.eval()
  images_export, labels_export = None, None
  for i, (images, labels) in enumerate(test_loader):
    if images_export is None or labels_export is None:
      images_export = images.data.numpy()
      labels_export = labels.data.numpy()
    else:
      images_export = np.concatenate((images_export, images.data.numpy()), axis=0)
      labels_export = np.concatenate((labels_export, labels.data.numpy()), axis=0)
    images, labels = apply_cuda(images), apply_cuda(labels)
    logits = model(images)
    _, pred = torch.max(logits.data, 1)
    if i == 0:
      print(images[0])
      print(logits[0], pred[0], labels[0])
    bs_ = labels.data.size()[0]
    match_count = (pred == labels.data).sum()
    accuracy = float(match_count)/float(bs_)
    print(datetime.now(), 'batch {}/{} with shape={}, accuracy={:.4f}'.format(
        i+1, n_batches_test, images.shape, accuracy))
    avg_acc += accuracy/float(n_batches_test)
  print(datetime.now(), 'test results: acc={:.4f}'.format(avg_acc))
  print(datetime.now(), 'total batch to be exported with shape={}'.format(
      images_export.shape))
  export_test_data_to_numpy(images_export, labels_export, data_folder)

if __name__ == '__main__':
  from config import lr, model_name, ckpt_name
  test(
      model_name=model_name,
      model_ckpt=ckpt_name+'.pth',
      dataset_name='sdd',
      data_folder='data')

