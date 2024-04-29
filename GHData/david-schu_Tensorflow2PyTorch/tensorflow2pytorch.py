import torch
from torchvision import datasets, transforms
import torchModel
import eval
from tfModels import tfModel

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print('Restore Tensorflow Model...')
sess = tf.Session()
tf_model = tfModel.Model()
checkpoint = tf.train.latest_checkpoint('./tfModels/adv_trained')
restorer = tf.train.Saver()
restorer.restore(sess, checkpoint)

print('Get trainable variable names and shapes: ')
for val in tf.trainable_variables():
    print(val.name + ': shape=' + str(val.shape))

print('Assign weights and biases...')
model_madry = torchModel.madry()
weights_cv1 = torch.from_numpy(sess.run('Variable:0')).permute((3, 2, 0, 1))
bias_cv1 = torch.from_numpy(sess.run('Variable_1:0'))
model_madry.conv1.weight = torch.nn.Parameter(weights_cv1)
model_madry.conv1.bias = torch.nn.Parameter(bias_cv1)

weights_cv2 = torch.from_numpy(sess.run('Variable_2:0')).permute((3, 2, 0, 1))
bias_cv2 = torch.from_numpy(sess.run('Variable_3:0'))
model_madry.conv2.weight = torch.nn.Parameter(weights_cv2)
model_madry.conv2.bias = torch.nn.Parameter(bias_cv2)

weights_fc1 = torch.from_numpy(sess.run('Variable_4:0')).T
bias_fc1 = torch.from_numpy(sess.run('Variable_5:0'))
model_madry.fc1.weight = torch.nn.Parameter(weights_fc1)
model_madry.fc1.bias = torch.nn.Parameter(bias_fc1)

weights_fc2 = torch.from_numpy(sess.run('Variable_6:0')).T
bias_fc2 = torch.from_numpy(sess.run('Variable_7:0'))
model_madry.fc2.weight = torch.nn.Parameter(weights_fc2)
model_madry.fc2.bias = torch.nn.Parameter(bias_fc2)

print('Test model:')
batch_size = 50
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size)

eval.evalClean(model_madry, test_loader)
print('Done!')