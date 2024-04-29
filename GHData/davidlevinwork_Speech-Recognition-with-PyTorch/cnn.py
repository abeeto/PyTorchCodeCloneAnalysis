import time as _time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as t
from gcommand_dataset import *
import torch.nn.functional as F

EPOCHS = 10
NUMBER_OF_LABELS = 30
LEARNING_RATE = 0.0001
OUTPUT_FILE_NAME = "test_y"


# <editor-fold desc="Model">

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.epoch = EPOCHS
        self.lr = LEARNING_RATE

        # First CNN layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second CNN layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third CNN layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth CNN layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth CNN layer
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # First linear layer
        self.linear6 = nn.Linear(512, 128)
        self.bn6 = nn.BatchNorm1d(128)

        # Second linear layer
        self.linear7 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.bn6(self.linear6(x)))

        x = self.linear7(x)
        return F.log_softmax(x, dim=1)

# </editor-fold>


# <editor-fold desc="Helpers">

# Function role: Initiate the data loaders.
def initLoaders(train, validation, test):
    train_loader = t.DataLoader(train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, sampler=None)
    valid_loader = t.DataLoader(validation, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, sampler=None)
    test_loader = t.DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, sampler=None)

    return train_loader, valid_loader, test_loader


# Function role: iterate the predictions, and return the correct & total values according to the true labels.
def setPredictionsValues(predictions, labels, total, correct):
    for index, pred in enumerate(predictions):
        total += 1
        if labels[index] == torch.argmax(pred):
            correct += 1
    return total, correct


# Function role is to print the given predictions to the output file.
def writeToFile(predictions, classes, files):
    file = open(OUTPUT_FILE_NAME, "w")
    last_line = predictions.shape[0] - 1
    for index, (pred, f) in enumerate(zip(predictions, files)):
        file.write(f'{f},{classes[pred]}')
        if index != last_line:
            file.write("\n")
    file.close()


def get_file_names(val):
    return [file[0].split('\\')[-1] or file[0].split('/')[-1] for file in val]

# </editor-fold>


# <editor-fold desc="Model's Validation Functions">

# Function role: run all the validation process (initiating the model --> train the model --> test the model).
def runValidationProcess(train_loader, valid_loader, test_loader):
    model = Model()
    trainWithValidation(model, train_loader, valid_loader)
    testValidation(model, test_loader)


def trainWithValidation(model, training_loader, validation_loader):
    # Set the optimizer regarding the model
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    # Initiate helpers
    valid_accuracy_arr, valid_loss_arr = [], []
    train_accuracy_arr, train_loss_arr = [], []

    for epoch in range(EPOCHS):
        loss_train = total_train = correct_train = 0
        # Set the model status as train mode
        model.train()

        for batch_idx, (data, labels) in enumerate(training_loader):
            # Init the gradients values
            optimizer.zero_grad()
            # Execute the forward propagation. The result size is (#Batch, #NumOfLabels)
            output = model(data)

            # Update the result of the current batch
            total_train, correct_train = setPredictionsValues(output, labels, total_train, correct_train)

            # Update the loss value (entropy loss is the preferred way for working with large scales of labels)
            entropy_loss = nn.CrossEntropyLoss()
            loss = entropy_loss(output, labels)
            # Execute the back propagation & save the gradients values
            loss.backward()
            # Update the model parameters according to the optimizer
            optimizer.step()
            # Update the loss value
            loss_train += loss.item()
        # Sum up the train results
        train_accuracy = round((correct_train / total_train), 3)
        train_loss_accuracy = round((loss_train / len(training_loader)), 3)
        train_accuracy_arr.append(train_accuracy), train_loss_arr.append(train_loss_accuracy)

        with torch.no_grad():
            validation_accuracy, validation_loss_accuracy = testValidation(model, validation_loader)
            valid_accuracy_arr.append(validation_accuracy), valid_loss_arr.append(validation_loss_accuracy)

        print(f"Epoch [{epoch}] | Train accuracy [{train_accuracy}] | Validation accuracy [{validation_accuracy}]")
        print(f"Epoch [{epoch}] | Train loss [{train_loss_accuracy}] | Validation loss [{validation_loss_accuracy}]")
        print("----------------------------------------------------------------")


def testValidation(model, loader):
    model.eval()
    # Calculate without saving the "calculation graph" (it will not affect the results)
    with torch.no_grad():
        loss_test = total_test = correct_test = 0
        for batch_idx, (data, labels) in enumerate(loader):
            # Execute the forward propagation. The result size is (#Batch, #NumOfLabels)
            output = model(data)
            # Update the loss value (entropy loss is the preferred way for working with large scales of labels)
            entropy_loss = nn.CrossEntropyLoss()
            loss = entropy_loss((output), (labels))
            loss_test += loss.item()
            # Update the result of the current batch
            total_test, correct_test = setPredictionsValues(output, labels, total_test, correct_test)
        # Sum up the validation results
        valid_accuracy = round((correct_test / total_test), 3)
        valid_loss_accuracy = round((loss_test / len(loader)), 3)
    return valid_accuracy, valid_loss_accuracy


# </editor-fold>


# <editor-fold desc="Model's Functions">

def train(model, loader):
    # Set the optimizer regarding the model
    optimizer = optim.Adam(model.parameters(), lr=model.lr)

    for epoch in range(EPOCHS):
        # Set the model status as train mode
        model.train()
        for batch_idx, (data, labels) in enumerate(loader):
            # Init the gradients values
            optimizer.zero_grad()
            # Execute the forward propagation. The result size is (#Batch, #NumOfLabels)
            output = model(data)
            # Calculate the loss function (using negative log likelihood)
            loss = F.nll_loss(output, labels)
            # Execute the back propagation & save the gradients values
            loss.backward()
            # Update the model parameters according to the optimizer
            optimizer.step()


def test(model, loader):
    predictions = []
    model.eval()
    # Calculate without saving the "calculation graph" (it will not affect the results)
    with torch.no_grad():
        for data, _ in loader:
            # Execute the forward propagation. The result size is (#Batch, #NumOfLabels)
            output = model(data)
            # Get the model's predictions for a given test set
            for index, label in enumerate(output):
                predictions.append(torch.argmax(label).item())
    predictions = np.asarray(predictions).astype(int)
    return predictions


# </editor-fold>


if __name__ == "__main__":
    start = _time.time()
    print("Start...")

    # Load the given data sets: train & validation & test
    train_data = GCommandLoader('data/train')
    valid_data = GCommandLoader('data/valid')
    test_data = GCommandLoader('data/test')

    # Initiate helpers
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    files = get_file_names(test_data.spects)

    # Initiate the loaders
    train_loader, valid_loader, test_loader = initLoaders(train_data, valid_data, test_data)

    # Uncomment this line if you want to see the model results via the validation process
    # runValidationProcess(train_loader, valid_loader, test_loader)

    # Run the model directly (without validation)
    model = Model()
    train(model, train_loader)
    predictions = test(model, test_loader)
    writeToFile(predictions, idx_to_class, files)

    end = _time.time()
    print(f'Total time: {end - start}')
    print("Finish...")
