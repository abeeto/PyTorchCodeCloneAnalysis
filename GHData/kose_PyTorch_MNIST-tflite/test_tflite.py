import torch
import tensorflow as tf
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import confusion_matrix

##
## test main function
##
def test(interpreter, dataset):

    # set in/out
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()[0]['index']

    # result
    labels_ground_truth = np.array([], dtype=np.int8)
    labels_pred = np.array([], dtype=np.int8)
    
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for data, label in testloader:

        # truth
        labels_ground_truth = np.append(labels_ground_truth, label.numpy())

        # set input tensor
        interpreter.set_tensor(input_details, data.numpy())

        # run
        interpreter.invoke()

        # get outpu tensor
        probs = interpreter.get_tensor(output_details)

        # predict label
        labels_pred = np.append(labels_pred, np.array(np.argmax(probs)))
        
    result = confusion_matrix(labels_ground_truth, labels_pred)

    print(result)


def main():

    #
    # test dataset
    #
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('data', train=False, transform=transform)

    #
    # model
    #
    modelname = "mnist_cnn.tflite"
    interpreter = tf.lite.Interpreter(model_path=modelname)

    #
    # testing
    #
    test(interpreter, dataset)


if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
