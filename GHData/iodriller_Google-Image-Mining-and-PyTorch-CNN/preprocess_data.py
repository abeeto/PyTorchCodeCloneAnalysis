from google_image_search import google_image_search
from prep_data import prep_data
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from neural_net import Net
from tqdm import tqdm


class preprocess_data():
    def __init__(
            self,
            IMG_SIZE=50):
        self.IMG_SIZE = IMG_SIZE

    def run(self, search_terms, DRIVER_PATH, no_of_images=5, validation_fraction=0.1, batch_size=50, epochs=3):
        for search_term in search_terms:
            google_image_search().search_and_download(search_term, DRIVER_PATH, number_images=no_of_images)
        prep_data().make_training_data()
        training_data = np.load("training_data.npy", allow_pickle=True)

        X = torch.Tensor([i[0] for i in training_data]).view(-1, self.IMG_SIZE, self.IMG_SIZE)
        X = X / np.max(np.ravel(X))
        y = torch.Tensor([i[1] for i in training_data])

        net = Net(IMG_SIZE=self.IMG_SIZE, category_amount=len(search_terms))

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        #  reserve X% of our data for validation
        val_size = int(len(X) * validation_fraction)

        train_X = X[:-val_size]
        train_y = y[:-val_size]

        test_X = X[-val_size:]
        test_y = y[-val_size:]

        print("Running the CNN")
        for epoch in range(epochs):
            for i in tqdm(range(0, len(train_X),
                                batch_size)):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev

                batch_X = train_X[i:i + batch_size].view(-1, 1, self.IMG_SIZE, self.IMG_SIZE)
                batch_y = train_y[i:i + batch_size]

                net.zero_grad()

                outputs = net(batch_X)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()  # Does the update
            try:
                print(f"Epoch: {epoch}. Loss: {loss}")
            except Exception as e:
                print(e, "couldn't calculate the loss")
                pass

        self.estimate_accuracy(test_X, test_y, net)
        return net

    def estimate_accuracy(self, test_X, test_y, net):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(test_X))):
                real_class = torch.argmax(test_y[i])
                net_out = net(test_X[i].view(-1, 1, self.IMG_SIZE, self.IMG_SIZE))[0]  # returns a list,
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1
        print("Accuracy: ", round(correct / total, 3))
