import argparse
import model_builder
import os
import shutil
import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from torchviz import make_dot
from tqdm import tqdm
from datetime import datetime


class NeuralNetwork(nn.Module):
    """
    The neural network to train for the MNIST dataset.
    """

    def __init__(self):
        """
        Set up the neural network loading in parameters defined in 'model_builder.py'.
        """
        super().__init__()
        # Load in defined parameters.
        self.layers = model_builder.define_layers()
        self.loss = model_builder.define_loss()
        self.optimizer = model_builder.define_optimizer(self)
        # Run on GPU if available.
        self.to(get_processing_device())

    def forward(self, image):
        """
        Feed forward an MNIST image into the neural network.
        :param image: The MNIST image as a proper tensor.
        :return: The final output layer from the network.
        """
        return self.layers(image)

    def predict(self, image):
        """
        Get the network's prediction for an MNIST image.
        :param image: The MNIST image as a proper tensor.
        :return: The number the network predicts for this image.
        """
        with torch.no_grad():
            # Get the highest confidence output value.
            return torch.argmax(self.forward(image), axis=-1)

    def optimize(self, image, label):
        """
        Optimize the neural network to fit the MNIST training data.
        :param image: The MNIST image as a proper tensor.
        :param label: The label of the MNIST image.
        :return: The network's loss on this prediction.
        """
        self.optimizer.zero_grad()
        loss = self.loss(self.forward(image), label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def get_processing_device():
    """
    Get the device to use for training, so we can use the GPU if CUDA is available.
    :return: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(tensor, device=get_processing_device()):
    """
    Convert an image to a tensor to run on the given device.
    :param tensor: The data to convert to a tensor.
    :param device: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    :return: The data ready to be used.
    """
    return tensor.to(device)


def mnist_details(title: str, dataset):
    """
    Output MNIST details to the console.
    :param title: The title of the dataset to tell if this is the training or testing dataset.
    :param dataset: The MNIST dataset.
    :return: Nothing.
    """
    labels = {}
    # Initialize all class indexes at zero.
    for i in range(len(dataset.classes)):
        labels[dataset.classes[i]] = 0
    # Count all labels.
    for i in range(len(dataset)):
        labels[dataset.classes[dataset[i][1]]] += 1
    print(f"{title} Dataset = {len(dataset)} Images")
    for label in labels:
        print(f"{label[0]}\t{labels[label]: >5}\t{labels[label] / len(dataset) * 100}%")


def data_image(dataloader, writer, title: str):
    """
    Generate a sample grid image of MNIST data.
    :param dataloader: An MNIST dataloader.
    :param writer: The TensorBoard writer.
    :param title: The title to give the image.
    :return: Nothing.
    """
    writer.add_image(title, torchvision.utils.make_grid(iter(dataloader).next()[0]))


def test(model, batch: int, dataloader):
    """
    Test a neural network.
    :param model: The neural network.
    :param batch: The batch size.
    :param dataloader: The MNIST dataloader which should be of the test dataset.
    :return: The model's accuracy.
    """
    # Count how many are correct.
    correct = 0
    # Loop through all data.
    for raw_image, raw_label in dataloader:
        image, label = to_tensor(raw_image), to_tensor(raw_label)
        # If properly predicted, count it as correct.
        correct += (label == model.predict(image)).sum()
    # Calculate the overall accuracy.
    return correct / (len(dataloader) * batch) * 100


def parameter_text_generator(accuracy):
    # Create text file to store parameter data.
    f = open(f"{os.getcwd()}/ModelParameters/" + a["name"] + ".txt", "w")
    # Write parameters to text file
    f.write("Accuracy Rate & Device: " + str(accuracy) + "\n" +
            "Name of the model: " + str(a["name"]) + "\n" +
            "Number of training epochs: " + str(a["epoch"]) + "\n" +
            "Training and testing batch size: " + str(a["batch"]) + "\n" +
            "Count image labels in the MNIST dataset: " + str(a["count"]) + "\n" +
            "Minimum rescaling size of training data: " + str(a["min"]) + "\n" +
            "Maximum rescaling size of training data: " + str(a["max"]) + "\n" +
            "Maximum rotation of training data: " + str(a["rot"]) + "\n")
    # Timestamp for text file.
    f.write(str(datetime.now()) + "\n")
    # Writes whether the model was loaded or not to text file.
    if a["load"]:
        f.write("\n Name of Model Loaded : " + str(a["load"]))
    else:
        f.write("Model is an original.")
    f.close()


def main(name: str, epochs: int, batch: int, count: bool, load: bool, mini: float, maxi: float, rot: float, att: int):
    """
    Main program execution.
    :param name: The name of the model to save files under.
    :param epochs: The number of training epochs.
    :param batch: The batch size.
    :param count: Whether to count MNIST data details or not.
    :param load: Whether to load an existing model or train a new one.
    :param mini: The minimum random scaling of the training dataset.
    :param maxi: The maximum random scaling of the training dataset.
    :param rot: The maximum random rotation of the training dataset.
    :param att: The amount of times to try the training.
    :return: Nothing.
    """
    print(f"MNIST Deep Learning\nRun 'tensorboard --logdir=runs' in '{os.getcwd()}' to see data.")
    if name == "Current":
        print("Cannot name the model 'Current' as that name is used for the current TensorBoard run.")
        return
    print(f"Running on GPU with CUDA {torch.version.cuda}." if torch.cuda.is_available() else "Running on CPU.")
    if load or att < 1:
        att = 1
    for attempt in range(0, att):
        if att > 1:
            print(f"Training attempt {attempt + 1} of {att}.")
        # Setup datasets.
        training_data = MNIST(
            root=os.getcwd(),
            train=True,
            download=True,
            transform=transforms.Compose([
                # Randomly zoom in on training data to add more variety.
                transforms.RandomResizedCrop(28, (mini, maxi)),
                # Randomly rotate the training data to add more variety.
                transforms.RandomRotation(rot),
                transforms.ToTensor()
            ])
        )
        testing_data = MNIST(
            root=os.getcwd(),
            train=False,
            download=True,
            transform=transforms.Compose([
                # Do not modify the testing data.
                transforms.ToTensor()
            ])
        )
        # Count MNIST details if set to.
        if count and a == 0:
            print("Totaling MNIST Dataset Details...")
            mnist_details("Training", training_data)
            mnist_details("Testing", testing_data)
        writer = SummaryWriter("runs/Current")
        no_model = True
        if load:
            # If a model does not exist to load decide to generate a new model instead.
            if not os.path.exists(f"{os.getcwd()}/Models/{name}.pt"):
                print(f"Model '{name}.pt' does not exist to load, will build new model instead.")
            else:
                try:
                    print(f"Loading '{name}'...")
                    model = NeuralNetwork()
                    model.load_state_dict(torch.load(f"{os.getcwd()}/Models/{name}.pt"))
                    print(f"Loaded '{name}'.")
                    no_model = False
                except:
                    print("Model to load has different structure 'model_builder'.py, cannot load, building new model.")
        if no_model:
            model = NeuralNetwork()
        dataloader = DataLoader(training_data, batch_size=batch, shuffle=True)
        print(len(dataloader.dataset))
        # Generate a sample image of the training data.
        data_image(dataloader, writer, 'Training Sample')
        if no_model and epochs > 0:
            # Train if we are building a new model, this is skipped if a model was loaded.
            print(f"Training '{name}' for {epochs} epochs with batch size {batch}...")
            # Train for the set number of epochs
            for epoch in range(epochs):
                msg = f"Epoch {epoch + 1}/{epochs} | Loss = " + (f"{loss / len(dataloader):.4}" if epoch > 0 else "N/A")
                # Reset loss every epoch.
                loss = 0
                for raw_image, raw_label in tqdm(dataloader, msg):
                    image, label = to_tensor(raw_image), to_tensor(raw_label)
                    loss += model.optimize(image, label)
                writer.add_scalar('Training Loss', loss / len(dataloader), epoch)
            print(f"Training '{name}' complete - Final loss = {loss / len(dataloader):.4}")
        dataloader = DataLoader(testing_data, batch_size=batch, shuffle=True)
        # Generate a sample image of the testing data.
        data_image(dataloader, writer, 'Testing Sample')
        # Test the model on the testing data.
        accuracy = test(model, batch, dataloader)
        writer.add_text(f"Accuracy", f"{accuracy}%")
        summary(model, input_size=(1, 28, 28))
        print(f"Accuracy = {accuracy}%")
        # Ensure folder to save models exists.
        if not os.path.exists(f"{os.getcwd()}/Models"):
            os.mkdir(f"{os.getcwd()}/Models")
        # Check if an existing model of the same name exists.
        if not load and os.path.exists(f"{os.getcwd()}/Models/{name}.pt"):
            print(f"Model '{name}.pt' already exists, loading and checking if new model is better...")
            try:
                existing = NeuralNetwork()
                existing.load_state_dict(torch.load(f"{os.getcwd()}/Models/{name}.pt"))
                existing_accuracy = test(existing, batch, dataloader)
                # If the existing model of the same name has better accuracy, don't save the new one.
                if existing_accuracy >= accuracy:
                    print(f"Existing '{name}.pt' has better accuracy score of {existing_accuracy}%, not saving new model.")
                    writer.close()
                    shutil.rmtree(f"{os.getcwd()}/runs/Current")
                    continue
                print(f"Existing '{name}.pt' has worse accuracy score of {existing_accuracy}%, overwriting with new model.")
            except:
                print("Model to load has different structure 'model_builder'.py, cannot load, will save the new model.")
        # Save the model.
        torch.save(model.state_dict(), f"{os.getcwd()}/Models/{name}.pt")
        # Create a graph of the model.
        img = to_tensor(iter(dataloader).next()[0])
        writer.add_graph(model, img)
        # Ensure the folder to save graph images exists.
        if not os.path.exists(f"{os.getcwd()}/Graphs"):
            os.mkdir(f"{os.getcwd()}/Graphs")
        try:
            y = model.forward(img)
            make_dot(y.mean(), params=dict(model.named_parameters())).render(f"{os.getcwd()}/Graphs/{name}", format="png")
            # Makes redundant file, remove it.
            os.remove(f"{os.getcwd()}/Graphs/{name}")
        except:
            "Could not generate graph image, make sure you have 'Graphviz' installed."
        writer.close()
        # If there is an older, worse run data with the same name, overwrite it.
        if os.path.exists(f"{os.getcwd()}/runs/{name}"):
            shutil.rmtree(f"{os.getcwd()}/runs/{name}")
        os.rename(f"{os.getcwd()}/runs/Current", f"{os.getcwd()}/runs/{name}")
        # Ensure Folder to save parameters exists
        if not os.path.exists(f"{os.getcwd()}/ModelParameters"):
            os.mkdir(f"{os.getcwd()}/ModelParameters")
        # Save model parameters and accuracy.
        parameter_text_generator({accuracy})


if __name__ == '__main__':
    desc = "MNIST Deep Learning\n-------------------\nTrain and test models on the MNIST dataset."
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument("-n", "--name", type=str, help="Name of the model.", default='Model')
    parser.add_argument("-e", "--epoch", type=int, help="Number of training epochs.", default=5)
    parser.add_argument("-b", "--batch", type=int, help="Training and testing batch size.", default=64)
    parser.add_argument("-l", "--load", help="Load the model with the given name and test it.", action="store_true")
    parser.add_argument("-c", "--count", help="Count image labels in the MNIST dataset.", action="store_true")
    parser.add_argument("-a", "--attempts", type=int, help="The amount of times to try the training.", default=1)
    parser.add_argument("--min", type=float, help="Minimum rescaling size of training data.", default=1)
    parser.add_argument("--max", type=float, help="Maximum rescaling size of training data.", default=1)
    parser.add_argument("-r", "--rot", type=float, help="Maximum rotation of training data.", default=0)
    a = vars(parser.parse_args())
    main(a["name"], a["epoch"], a["batch"], a["count"], a["load"], a["min"], a["max"], a["rot"], a["attempts"])
