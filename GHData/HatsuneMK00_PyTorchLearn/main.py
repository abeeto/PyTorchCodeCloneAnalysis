
from gtsrb.cnn_model_small_2 import SmallCNNModel2
from torchinfo import summary

if __name__ == '__main__':
    model = SmallCNNModel2()
    summary(model, input_size=(1, 3, 32, 32))