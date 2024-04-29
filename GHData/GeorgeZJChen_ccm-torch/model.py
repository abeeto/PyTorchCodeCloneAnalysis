from __future__ import print_function, division
import torch
import pickle
import os
import random
from threading import Thread

def conv(kernel_size, in_channels, filters, padding=(1,1), strides=(1,1), dilation=1, name=None, act=None, dropout=None):
  if kernel_size==1:
    padding=0
  layers = []
  layers.append(torch.nn.Conv2d(in_channels, filters, kernel_size, strides, padding, dilation))
  if dropout is not None:
    layers.append(torch.nn.Dropout(dropout))
  if act is not None:
    layers.append(act)
  return torch.nn.Sequential(*layers)
def conv_t(kernel_size, in_channels, filters, strides=2, padding=1, act=None, dropout=None, training=True):
  layers = []
  layers.append(torch.nn.ConvTranspose2d(in_channels, filters, kernel_size, strides, padding=padding, output_padding=0))
  if dropout is not None:
    layers.append(torch.nn.Dropout(dropout))
  if act is not None:
    layers.append(act)
  return torch.nn.Sequential(*layers)
def conv_layer(conv_op, input, act=None, dropout=None):
  out = conv_op(input)
  if act is not None:
    out = act(out)
  if dropout is not None:
    out = torch.nn.functional.dropout(out, p=dropout)
  return out
def maxpool(kernel_size, strides=2):
  return torch.nn.MaxPool2d(kernel_size, strides)
def concat(tensors, axis=1):
  return torch.cat(tensors, axis)
def leaky_relu(input):
  return torch.nn.functional.leaky_relu_(input, 0.2)
def abs_loss(predict, target):
  loss = torch.abs(predict - target)
  loss = torch.mean(loss)
  return loss
class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    layers = []
    layers.append(conv(3, 3, 64, act=torch.nn.ReLU()))
    layers.append(conv(3, 64, 64, act=torch.nn.ReLU()))
    layers.append(maxpool(2))
    layers.append(conv(3, 64, 128, act=torch.nn.ReLU()))
    layers.append(conv(3, 128, 128, act=torch.nn.ReLU()))
    layers.append(maxpool(2))
    layers.append(conv(3, 128, 256, act=torch.nn.ReLU()))
    layers.append(conv(3, 256, 256, act=torch.nn.ReLU()))
    layers.append(conv(3, 256, 256, act=torch.nn.ReLU()))
    layers.append(maxpool(2))
    layers.append(conv(3, 256, 512, act=torch.nn.ReLU()))
    layers.append(conv(3, 512, 512, act=torch.nn.ReLU()))
    layers.append(conv(3, 512, 512, act=torch.nn.ReLU()))
    self.vgg = torch.nn.Sequential(*layers)

    self.layer10 = conv(3, 512, 256)

    self.layer11_0 = conv(3, 256, 512, strides=2)
    self.layer11 = conv(3, 512, 256)

    self.layer12_0 = conv(3, 256, 512, strides=2)
    self.layer12 = conv(3, 512, 256)

    self.layer13_0 = conv(3, 256, 512, strides=2)
    self.layer13 = conv(3, 512, 256)

    self.layer14_0 = conv(3, 256, 512, strides=2)
    self.layer14 = conv(3, 512, 256)

    self.layer15 = conv((3,4), 256, 1024, padding=0)


    self.out15 = conv(1, 1024, 1)

    self.out14_0 = conv_t((3,4), 1024, 256, padding=0)
    self.out14 = conv(1, 512, 1)

    self.out13_0 = conv_t((3,4), 1024, 256, padding=0)
    self.out13_1 = conv_t(4, 512, 256)
    self.out13 = conv(1, 512, 1)

    self.out12_0 = conv_t((3,4), 1024, 256, padding=0)
    self.out12_1 = conv_t(4, 512, 256)
    self.out12_2 = conv_t(4, 512, 256)
    self.out12 = conv(1, 512, 1)

    self.out11_0 = conv_t((3,4), 1024, 256, padding=0)
    self.out11_1 = conv_t(4, 512, 256)
    self.out11_2 = conv_t(4, 512, 256)
    self.out11_3 = conv_t(4, 512, 256)
    self.out11 = conv(1, 512, 1)

    self.out10_0 = conv_t((3,4), 1024, 256, padding=0)
    self.out10_1 = conv_t(4, 512, 256)
    self.out10_2 = conv_t(4, 512, 256)
    self.out10_3 = conv_t(4, 512, 256)
    self.out10_4 = conv_t(4, 512, 256)
    self.out10 = conv(1, 512, 1)

    self.optimizer = torch.optim.SGD([param for name, param in self.named_parameters() if 'vgg' not in name],
            lr=1e-4, momentum=0.9, weight_decay=1e-4)
    self.vgg_optimizer = torch.optim.SGD(self.vgg.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)

    for name, param in self.named_parameters():
      if param.requires_grad:
        if 'weight' in name:
          torch.nn.init.xavier_uniform_(param)
        if 'bias' in name:
          torch.nn.init.constant_(param, 0.0)

  def forward(self, input, dropout=0):

    layer10 = conv_layer(self.layer10, self.vgg(input), act=leaky_relu, dropout=dropout)

    layer11 = conv_layer(self.layer11_0, layer10, act=leaky_relu, dropout=dropout)
    layer11 = conv_layer(self.layer11, layer11, act=leaky_relu, dropout=dropout)

    layer12 = conv_layer(self.layer12_0, layer11, act=leaky_relu, dropout=dropout)
    layer12 = conv_layer(self.layer12, layer12, act=leaky_relu, dropout=dropout)

    layer13 = conv_layer(self.layer13_0, layer12, act=leaky_relu, dropout=dropout)
    layer13 = conv_layer(self.layer13, layer13, act=leaky_relu, dropout=dropout)

    layer14 = conv_layer(self.layer14_0, layer13, act=leaky_relu, dropout=dropout)
    layer14 = conv_layer(self.layer14, layer14, act=leaky_relu, dropout=dropout)

    layer15 = conv_layer(self.layer15, layer14, act=leaky_relu, dropout=0)

    out15 = conv_layer(self.out15, layer15, act=leaky_relu)

    layer = conv_layer(self.out14_0, layer15, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer14])
    out14 = conv_layer(self.out14, layer, act=leaky_relu)

    layer = conv_layer(self.out13_0, layer15, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer14])
    layer = conv_layer(self.out13_1, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer13])
    out13 = conv_layer(self.out13, layer, act=leaky_relu)

    layer = conv_layer(self.out12_0, layer15, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer14])
    layer = conv_layer(self.out12_1, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer13])
    layer = conv_layer(self.out12_2, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer12])
    out12 = conv_layer(self.out12, layer, act=leaky_relu)

    layer = conv_layer(self.out11_0, layer15, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer14])
    layer = conv_layer(self.out11_1, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer13])
    layer = conv_layer(self.out11_2, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer12])
    layer = conv_layer(self.out11_3, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer11])
    out11 = conv_layer(self.out11, layer, act=leaky_relu)

    layer = conv_layer(self.out10_0, layer15, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer14])
    layer = conv_layer(self.out10_1, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer13])
    layer = conv_layer(self.out10_2, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer12])
    layer = conv_layer(self.out10_3, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer11])
    layer = conv_layer(self.out10_4, layer, act=leaky_relu, dropout=dropout)
    layer = concat([layer, layer10])
    out10 = conv_layer(self.out10, layer, act=leaky_relu)

    train_outs = [out15, out14, out13, out12, out11, out10]
    monitored = None
    return train_outs, monitored

  def train(self, global_step, train_inputs, train_targets, learning_rate_scheduler):
    random_dropout = random.random()*0.3

    alpha = learning_rate_scheduler(global_step)
    alpha_vgg = alpha/2

    self.vgg_optimizer.param_groups[0]['lr'] = alpha_vgg
    self.optimizer.param_groups[0]['lr'] = alpha

    train_targets = [torch.from_numpy(target).float().cuda() for target in train_targets]
    train_inputs = torch.from_numpy(train_inputs).float().cuda().permute(0,3,1,2)

    self.vgg_optimizer.zero_grad()
    self.optimizer.zero_grad()

    train_outs, train_m = self.forward(train_inputs, random_dropout)
    train_loss = self.loss_fn(train_outs, train_targets)

    train_loss.backward()

    self.vgg_optimizer.step()
    self.optimizer.step()

    train_outs = [torch.nn.functional.relu(out).permute(0,2,3,1) for out in train_outs]
    return train_outs, train_loss, train_m

  def test(self, test_inputs, test_targets):

    test_targets = [torch.from_numpy(target).float().cuda() for target in test_targets]
    test_inputs = torch.from_numpy(test_inputs).float().cuda().permute(0,3,1,2)

    test_outs, _ = self.forward(test_inputs, 0)
    test_loss = self.loss_fn(test_outs, test_targets)

    test_outs = [torch.nn.functional.relu(out).permute(0,2,3,1) for out in test_outs]
    return test_outs, test_loss

  def loss_fn(self, outputs, targets):
    out15, out14, out13, out12, out11, out10 = outputs
    target15, target14, target13, target12, target11, target10 = targets
    loss = 0
    loss += abs_loss(out15, target15) / 16 / 12 * 10
    loss += abs_loss(out14, target14) / 16 * 2
    loss += abs_loss(out13, target13) / 4
    loss += abs_loss(out12, target12) * 1
    loss += abs_loss(out11, target11) * 4
    loss += abs_loss(out10, target10) * 16
    return loss


class Saver:
  def __init__(self, model, path='./', max_to_keep=1):
    if not path.endswith('/'):
      path += '/'
    if not os.path.exists(path):
      os.makedirs(path)
    self.model = model
    self.path = path
    self.max_to_keep = max_to_keep
    self.checkpoint_path = path+'checkpoint.pkl'
    if not os.path.exists(self.checkpoint_path):
      checkpoints = []
    else:
      checkpoints = self._read_checkpoints()
      if len(checkpoints) > max_to_keep:
        checkpoints = checkpoints[-max_to_keep:]
    self._write_checkpoints(checkpoints)
  def _write_checkpoints(self, checkpoints):
    def _write(checkpoints):
      with open(self.checkpoint_path, 'wb') as f:
        pickle.dump(checkpoints, f)
    thread = Thread(target=_write, args=(checkpoints,))
    thread.start()
    thread.join()
  def _read_checkpoints(self):
    try:
      with open(self.checkpoint_path, 'rb') as f:
        checkpoints = pickle.load(f)
    except EOFError:
      checkpoints = []
    return checkpoints
  def add_checkpoint(self, name):
    checkpoints = self._read_checkpoints()
    if len(checkpoints)==self.max_to_keep:
      name_to_delete = checkpoints.pop(0)
      self._delete_checkpoint(name_to_delete, checkpoints)
    checkpoints.append(name)
    self._write_checkpoints(checkpoints)
  def _delete_checkpoint(self, name_to_delete, checkpoints):
    if not name_to_delete in checkpoints and os.path.exists(self.path+name_to_delete):
      os.remove(self.path+name_to_delete)
  def last_checkpoint(self, n=-1):
    checkpoints = self._read_checkpoints()
    assert (n<0 and -n<=len(checkpoints)) or (n>=0 and n<len(checkpoints)-1), "Invalid checkpoint index: "+str(n)
    return checkpoints[n]
  def save(self, name, global_step):
    torch.save({
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            }, self.path+name)
    self.add_checkpoint(name)
  def restore(self, name):
    print('INFO: Restoring parameters from', self.path+name)
    checkpoint = torch.load(self.path+name)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['global_step']
