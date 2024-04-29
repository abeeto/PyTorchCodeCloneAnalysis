import os
import optparse
from mnist_nn import RunMnistNN
from mnist_nn import MnistNNPlot
from mnist_nn import MnistNNLoad
from mnist_parse import ParseFile
from mnist_nn import MnistNNParameter

from mnist_cnn import RunMnistCNN
from mnist_cnn import MnistCNNPlot
from mnist_cnn import MnistCNNLoad
from mnist_cnn import MnistConvImage
from mnist_cnn import MnistCNNParameter

def parse_file():
    train_img = "./MNIST/train-images-idx3-ubyte"
    train_label = "./MNIST/train-labels-idx1-ubyte"
    test_img = "./MNIST/t10k-images-idx3-ubyte"
    test_label = "./MNIST/t10k-labels-idx1-ubyte"
    ParseFile(train_img, train_label, 0).parse_image()
    ParseFile(train_img, train_label, 0).parse_label()
    ParseFile(test_img, test_label, 0).parse_image()
    ParseFile(test_img, test_label, 0).parse_label()

def classify_file():
    train_img = "./MNIST/train-images-idx3-ubyte"
    train_label = "./MNIST/train-labels-idx1-ubyte"
    test_img = "./MNIST/t10k-images-idx3-ubyte"
    test_label = "./MNIST/t10k-labels-idx1-ubyte"
    for i in range(0, 10):
        ParseFile(train_img, train_label, i).classify()
        ParseFile(test_img, test_label, i).classify()

def run_mnist_nn(nn_batchsize, nn_worker,
            nn_learning_rate, nn_momentum, 
            nn_cuda, nn_epoch_num):
    # Load data with mini-batch
    train_loader, test_loader = MnistNNLoad(nn_batchsize, 
                nn_worker).load_data()
    # Use Neural Network as training model
    model, criterion, optimizer = MnistNNParameter(
                nn_learning_rate,
                nn_momentum,
                nn_cuda).mnist_function()
    # Get experimental results in every epoch
    train_accuracy_rate = []
    test_accuracy_rate = []
    for epoch in range(1, nn_epoch_num + 1):
        train_loss, train_accuracy = RunMnistNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader,
                    nn_cuda).train_mnist_nn()
        test_loss, test_accuracy = RunMnistNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader,
                    nn_cuda).test_mnist_nn()
        train_accuracy_rate.append(train_accuracy)
        test_accuracy_rate.append(test_accuracy)
        print("Epoch:{:d}, Train Accuracy:{:7.6f}, Test Accuracy:{:7.6f}"
                    .format(epoch, train_accuracy, test_accuracy))
    # Make plot for Mnist-NN model
    MnistNNPlot(train_accuracy_rate,
                test_accuracy_rate,
                nn_epoch_num).plot()

def run_mnist_cnn(cnn_batchsize, cnn_worker,
            cnn_learning_rate, cnn_momentum, 
            cnn_cuda, cnn_epoch_num):
    # Load data with mini-batch
    train_loader, test_loader = MnistCNNLoad(cnn_batchsize, 
                cnn_worker).load_data()
    # Use Neural Network as training model
    model, criterion, optimizer = MnistCNNParameter(
                cnn_learning_rate,
                cnn_momentum,
                cnn_cuda).mnist_function()
    # Get experimental results in every epoch
    train_accuracy_rate = []
    test_accuracy_rate = []
    for epoch in range(1, cnn_epoch_num + 1):
        train_loss, train_accuracy = RunMnistCNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader,
                    cnn_cuda).train_mnist_cnn()
        test_loss, test_accuracy = RunMnistCNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader,
                    cnn_cuda).test_mnist_cnn()
        train_accuracy_rate.append(train_accuracy)
        test_accuracy_rate.append(test_accuracy)
        print("Epoch:{:d}, Train Accuracy:{:7.6f}, Test Accuracy:{:7.6f}"
                    .format(epoch, train_accuracy, test_accuracy))
    # Make plot for Mnist-CNN model
    MnistCNNPlot(train_accuracy_rate,
                test_accuracy_rate,
                cnn_epoch_num).plot()
    # Make plot for image after 1st conv
    cnn_batchsize = 1
    cnn_worker = 4
    _, one_loader = MnistCNNLoad(cnn_batchsize, cnn_worker).load_data()
    MnistConvImage(one_loader, cnn_cuda).show_conv_image()

if __name__ == "__main__":
    if os.path.isdir("./mnist") == False:
        parse_file()
    if os.path.isdir("./mnist_classified") == False:
        classify_file()
    # Mnist Neural Network parameters
    nn_batchsize = 640
    nn_worker = 2
    nn_learning_rate = 0.01
    nn_momentum = 0.5
    nn_cuda = True
    nn_epoch_num = 50
    # Mnist Conv Neural Network parameters
    cnn_batchsize = 640
    cnn_worker = 2
    cnn_learning_rate = 0.01
    cnn_momentum = 0.5
    cnn_cuda = True
    cnn_epoch_num = 2
    # Parse command line arguments
    parser = optparse.OptionParser()
    parser.add_option("--nn", action = "store_const",
                const = "nn", dest = "element", default = "nn")
    parser.add_option("--cnn", action = "store_const",
                const = "cnn", dest = "element")
    parser.add_option("--both", action = "store_const",
                const = "both", dest = "element")
    options, args = parser.parse_args()
    if options.element == "nn":
        run_mnist_nn(nn_batchsize, nn_worker, nn_learning_rate,
            nn_momentum, nn_cuda, nn_epoch_num)
    elif options.element == "cnn":
        run_mnist_cnn(cnn_batchsize, cnn_worker, cnn_learning_rate,
            cnn_momentum, cnn_cuda, cnn_epoch_num)
    elif options.element == "both":
        run_mnist_nn(nn_batchsize, nn_worker, nn_learning_rate,
            nn_momentum, nn_cuda, nn_epoch_num)
        run_mnist_cnn(cnn_batchsize, cnn_worker, cnn_learning_rate,
            cnn_momentum, cnn_cuda, cnn_epoch_num)