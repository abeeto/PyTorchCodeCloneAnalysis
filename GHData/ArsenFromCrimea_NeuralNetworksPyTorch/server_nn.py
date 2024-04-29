MONITOR_PORT_NUMBER = 5502

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = None
folder = 'data'
device = None
train_loader = None
criterion = None
acc = None
loss = None
batch_idx = None
lr = None
optimizer = None


# using_foreign_collection = False

# if using_foreign_collection:
#     from keras.datasets import cifar10
#
#     set_of_train_pictures, set_of_test_pictures = cifar10.load_data()
#
#     (cifar10_sources_train, cifar10_labels_train), (cifar10_sources_test, cifar10_labels_test) = (
#         set_of_train_pictures, set_of_test_pictures)
#     train_count = len(cifar10_sources_train)  # CIFAR10 only. I have to add MNIST
#     fill_state_dict("cifar10")
#
# else:
#     MyThread().start()

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Our_neural_network(nn.Module):

    # Свёрточный вариант
    def __init__(self, depth, count_of_classes):
        super(Our_neural_network, self).__init__()
        padding = 1
        pool = 2
        kernel_size = 2 * padding + 1
        filters1 = 32
        filters2 = 64
        # парочка свёрток с dawnsampling'ом:
        self.layer0 = nn.Conv2d(depth, filters1, kernel_size=kernel_size, padding=padding).to(device)
        self.layer1 = nn.ReLU().to(device)
        self.layer2 = nn.Conv2d(filters1, filters1, kernel_size=kernel_size, padding=0).to(device)
        self.layer3 = nn.ReLU().to(device)
        self.layer4 = nn.MaxPool2d(kernel_size=pool).to(device)
        self.layer5 = nn.Dropout2d(0.25).to(device)
        # ещё парочка свёрток с dawnsampling'ом:
        self.layer6 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size,
                                padding=padding).to(device)
        self.layer7 = nn.ReLU().to(device)
        self.layer8 = nn.Conv2d(filters2, filters2, kernel_size=kernel_size,
                                padding=0).to(device)
        self.layer9 = nn.ReLU().to(device)
        self.layer10 = nn.MaxPool2d(kernel_size=pool).to(device)
        self.layer11 = nn.Dropout2d(0.25).to(device)
        # ну и двуслойный перспептрон напоследок:
        self.layer12 = Flatten().to(device)
        inner_knots = 512
        self.layer13 = nn.Linear(
            ((input_height - 2) // 2 - 2) // 2 * (((input_width - 2) // 2 - 2) // 2) * filters2, inner_knots).to(device)
        self.layer14 = nn.ReLU().to(device)
        self.layer15 = nn.Dropout2d(0.5).to(device)
        self.layer16 = nn.Linear(inner_knots, count_of_classes).to(device)
        self.layer17 = nn.Softmax().to(device)
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                       self.layer7,
                       self.layer8, self.layer9, self.layer10, self.layer11,
                       self.layer12, self.layer13, self.layer14, self.layer15, self.layer16, self.layer17]

    # Примитивный (не свёрточный) вариант
    # def __init__(self, depth, count_of_classes):
    #     super(Our_neural_network, self).__init__()
    #     self.layer0 = Flatten()
    #     inner_knots1_3 = 200
    #     self.layer1 = nn.Linear(depth * input_height * input_width, inner_knots1_3)
    #     self.layer2 = nn.ReLU()
    #     inner_knots3_5 = 200
    #     self.layer3 = nn.Linear(inner_knots1_3, inner_knots3_5)
    #     self.layer4 = nn.ReLU()
    #     self.layer5 = nn.Linear(inner_knots3_5, count_of_classes)
    #     self.layer6 = nn.Softmax()
    #     self.layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]

    def forward(self, inputs):
        network_output = inputs
        for layer in self.layers:
            network_output = layer(network_output)
            # print("Shape: ", network_output.shape,", train count=",train_count)
        return network_output


import time

# def fill_state_dict(model_name):
#     import pickle
#     directory = "weights/python/" + model_name
#     state_dict = {}
#     from pathlib import Path
#     # пока я не научился определять количество файлов в папке,
#     # будем пользоваться тем, что количество слоёв вряд ли превосходит 50
#     for i in range(50):
#         layer_name = "layer" + str(i)
#         file_name_for_weight = layer_name + ".weight"
#         file_name_for_bias = layer_name + ".bias"
#         if Path(directory + "/" + file_name_for_weight).exists():
#             for_weight = open(directory + "/" + file_name_for_weight, "rb")
#             for_bias = open(directory + "/" + file_name_for_bias, "rb")
#             weight = pickle.load(for_weight)
#             print(file_name_for_weight, weight.shape)
#             bias = pickle.load(for_bias)
#             print(file_name_for_bias, bias.shape)
#             state_dict[file_name_for_weight] = torch.tensor(weight)
#             state_dict[file_name_for_bias] = torch.tensor(bias)
#     # print(state_dict)
#     our_neural_network.load_state_dict(state_dict)


from threading import Thread


class MyThread(Thread):
    def train(self):
        global epoch, loss, acc, batch_idx
        epoch = 0
        while True:
            if stop or self is not myThread:
                return
            correct = 0
            count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                if stop or self is not myThread:
                    return
                data, target = Variable(data.to(device)), Variable(target.to(device))
                # shape of data is (batch_size, depth, input_height, input_width)
                optimizer.zero_grad()

                net_out = our_neural_network(data)
                predictions = net_out.data.max(1)
                print(net_out.size(), target.size(), "class=",
                      predictions[1], "probability=", predictions[0])

                if criterion is nn.functional.cross_entropy:
                    loss_tensor = criterion(net_out, target)
                elif type(criterion) is nn.NLLLoss:
                    loss_tensor = criterion(net_out, target)  # Why?
                    print(type(loss_tensor), loss_tensor)
                else:
                    import numpy as np
                    classes = len(net_out[0])
                    one_hot_target = np.zeros((len(target), classes), float)
                    for t in range(len(target)):
                        for c in range(classes):
                            if c == target[t].item():
                                one_hot_target[t][c] = 1

                    loss_tensor = criterion.forward(net_out, torch.from_numpy(one_hot_target).to(device).float())

                pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()

                loss_tensor.backward()

                optimizer.step()
                count += len(data)
                loss = float(loss_tensor.item())
                print('Train Epoch: {} [{} from {}]\tLoss: {:.6f}'.format(
                    epoch, count, train_count,
                    loss))
                acc = float(correct.data) / count

            print('Accuracy: {}%'.format(round(100 * acc)))
            epoch += 1

    def run(self):
        global acc, loss, batch_idx, epoch, stop, our_neural_network
        while True:
            while stop:
                time.sleep(1)
            if self is not myThread:
                return
            self.train()


# print(our_neural_network.state_dict())


# def validate():
#     # run a test loop
#     test_loss = 0
#     correct = 0
#     import cv2
#
#     test_loader = torch.utils.data.DataLoader(
#         set_of_test_pictures,
#         batch_size=batch_size, shuffle=True)
#
#     if using_keras:
#         test_count = len(cifar10_sources_test)
#     else:
#         test_count = len(set_of_test_pictures)
#
#     print("Hello! Объём выборки ", test_count)
#
#     for source_id in range(test_count):
#         print("Testing. ", source_id, " from ", test_count)
#         if using_keras:
#             raw_picture = cifar10_sources_test[source_id]
#             target = cifar10_labels_test[source_id][0]
#         else:
#             raw_picture = set_of_test_pictures[source_id][0]
#             target = set_of_test_pictures[source_id][1]
#
#         target = torch.from_numpy(np.array([target]))
#
#         if using_keras:
#             picture = np.zeros((1, 3, input_height, input_width),
#                                dtype=np.float32)  # float32 only!
#             for ii in range(input_height):
#                 for jj in range(input_width):
#                     for c in range(3):
#                         picture[0][c][ii][jj] = raw_picture[ii][jj][c]
#
#             source = torch.from_numpy(picture)
#         else:
#             # for ii in range(input_height):
#             #     for jj in range(input_width):
#             #         for c in range(3):
#             #             picture[0][c][ii][jj] = raw_picture[c][ii][jj]
#             source = raw_picture.reshape(1, 3, input_height, input_width)
#
#         data, target = Variable(source, volatile=True), Variable(target)
#
#         net_out = our_neural_network(data)
#         # sum up batch loss
#         test_loss += criterion(net_out, target).data
#         pred = net_out.data.max(1)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data).sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
#         test_loss,
#         100. * correct / test_count))


# validate()


from http.server import BaseHTTPRequestHandler, HTTPServer


def to_json(value):
    if value is None:
        return "null"
    else:
        return str(value)


class MonitorHandler(BaseHTTPRequestHandler):

    def write(self, words):
        self.wfile.write(words.encode("UTF-8"))

    # Handler for the GET requests
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        global acc
        global loss
        global stop
        global train_loader
        global batch_size, lr, device, myThread, collection
        global our_neural_network, optimizer, criterion, input_width, input_height, depth, set_of_train_pictures, set_of_test_pictures, train_count
        if self.path == "/send_state_to_browser":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            self.write('{"batch":' + to_json(batch_idx) + ',"acc":' + to_json(acc) + ',"loss":' + to_json(loss) + '}')
            return
        if self.path.startswith("/load_monitor"):

            parsed_path = urlparse(self.path)

            set_of_train_pictures = collection(folder, train=True, download=True,
                                               transform=transform)
            set_of_test_pictures = collection(folder, train=True, download=False,
                                              transform=transform)
            train_count = len(set_of_train_pictures)
            myThread.start()
            print("Type of collection: ", type(set_of_train_pictures))

            batch_size = int(parse_qs(parsed_path.query)["batch_size"][0])
            train_loader = torch.utils.data.DataLoader(
                set_of_train_pictures,
                batch_size=batch_size, shuffle=True)
            if parse_qs(parsed_path.query)["criterion"][0] == "NLLLoss":
                criterion = nn.NLLLoss()
            elif parse_qs(parsed_path.query)["criterion"][0] == "functional.cross_entropy":
                criterion = nn.functional.cross_entropy
            elif parse_qs(parsed_path.query)["criterion"][0] == "L1Loss":
                criterion = nn.L1Loss()
            elif parse_qs(parsed_path.query)["criterion"][0] == "MSELoss":
                criterion = nn.MSELoss()

            lr = float(parse_qs(parsed_path.query)["lr"][0])
            optimizer_name = parse_qs(parsed_path.query)["optimizer"][0]
            if optimizer_name == "RMSprop":
                optimizer = torch.optim.RMSprop(our_neural_network.parameters(), lr=lr, weight_decay=1E-6)
            else:
                optimizer = torch.optim.SGD(our_neural_network.parameters(), lr=lr, momentum=0.9)

            stop = False
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            f = open("Watch_training.html", 'r')
            self.write(f.read())
            acc = "null"
            loss = "null"
            return
        if self.path.startswith("/load_launcher"):
            parsed_path = urlparse(self.path)
            if parse_qs(parsed_path.query)["model"][0] == "CIFAR10":
                input_width = 32
                input_height = 32
                depth = 3
                collection = datasets.CIFAR10
            else:
                input_width = 28
                input_height = 28
                depth = 1
                collection = datasets.MNIST
            device = torch.device(parse_qs(parsed_path.query)["device"][0])
            our_neural_network = Our_neural_network(depth, 10)
            # our_neural_network.to(device)
            stop = True
            myThread = MyThread()

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            f = open("Launcher.html", 'r')
            self.write(f.read())
            return
        if self.path == "/":
            stop = True
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            f = open("index.html", 'r')
            self.write(f.read())
            return
        if self.path == "/know_available_devices":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if torch.cuda.is_available():
                self.write('{"isCudaAvailable":true}')
            else:
                self.write('{"isCudaAvailable":false}')
            return
        if self.path.startswith("/change_learning_rate"):
            return
        if self.path == "/architecture":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.write("<html>")
            self.write("<head>")
            self.write("<title>Our architecture</title>")
            self.write("</head>")
            self.write("<body>")
            self.write("Architecture for <a href='/'>our neural network</a>:")
            self.write("<pre>")
            self.write(str(our_neural_network))
            self.write("</pre>")
            self.write("</body>")
            self.write("</html>")


monitor_server = HTTPServer(('', MONITOR_PORT_NUMBER), MonitorHandler)
print('I launch the monitor server on port ', MONITOR_PORT_NUMBER)

# Wait forever for incoming http requests
monitor_server.serve_forever()
