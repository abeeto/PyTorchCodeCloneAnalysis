# Before running, install required packages:
# pip install numpy torch torchvision pytorch-ignite tensorboardX tensorboard

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from datetime import datetime
from tensorboardX import SummaryWriter

def fake_data():
    # 4 images of shape 1x16x16 with labels 0, 1, 2, 3
    return [np.random.rand(4, 1, 16, 16), np.arange(4)]


# ----------------------------------- Setup -----------------------------------
# - images has array shape (num samples, channels, )
# - labels has array shape (num samples, )
train_data = fake_data()  # required
val_data = fake_data()    # optional
test_data = None          # optional

# Set up hyperparameters.
lr = 0.001
batch_size = 128
num_epochs = 5

# Set up logging.
experiment_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(logdir=f"logs/{experiment_id}")
print_every = 1  # batches

# Set up device.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ------------------------------- Preprocessing -------------------------------
def preprocess(data, name):
    if data is None:  # val/test can be empty
        return None

    images, labels = data

    # Rescale images to 0-255 and convert to uint8.
    # Note: This is done for each dataset individually, which is usually ok if all
    # datasets look similar. If not, scale all datasets based on min/ptp of train set.
    images = (images - np.min(images)) / np.ptp(images) * 255
    images = images.astype(np.uint8)

    # If images are grayscale, convert to RGB by duplicating channels.
    if images.shape[1] == 1:
        images = np.stack((images[:, 0],) * 3, axis=1)

    # Resize images and transform images torch tensor.
    images = images.transpose((0, 2, 3, 1))  # channels-last, required for transforms.ToPILImage
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = torch.stack(list(map(transform, images)))

    # Convert labels to tensors.
    labels = torch.from_numpy(labels).long()

    # Construct dataset.
    dataset = TensorDataset(images, labels)

    # Wrap in data loader.
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(name=="train"), **kwargs)
    return loader

train_loader = preprocess(train_data, "train")
val_loader = preprocess(val_data, "val")
test_loader = preprocess(test_data, "test")


# ----------------------------------- Model -----------------------------------
# Set up model, loss, optimizer.
model = models.alexnet(pretrained=True)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# --------------------------------- Training ----------------------------------
# Set up pytorch-ignite trainer and evaluator.
trainer = create_supervised_trainer(
    model,
    optimizer,
    loss_func,
    device=device,
)
metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(loss_func),
}
evaluator = create_supervised_evaluator(
    model, metrics=metrics, device=device
)

@trainer.on(Events.ITERATION_COMPLETED(every=print_every))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(
        f"Epoch {trainer.state.epoch} / {num_epochs}, "
        f"batch {batch} / {trainer.state.epoch_length}: "
        f"loss: {trainer.state.output:.3f}"
    )

@trainer.on(Events.EPOCH_COMPLETED)
def log_epoch(trainer):
    print(f"Epoch {trainer.state.epoch} / {num_epochs} average results: ")

    def log_results(name, metrics, epoch):
        print(
            f"{name + ':':6} loss: {metrics['loss']:.3f}, "
            f"accuracy: {metrics['accuracy']:.3f}"
        )
        writer.add_scalar(f"{name}_loss", metrics["loss"], epoch)
        writer.add_scalar(f"{name}_accuracy", metrics["accuracy"], epoch)

    # Train data.
    evaluator.run(train_loader)
    log_results("train", evaluator.state.metrics, trainer.state.epoch)

    # Val data.
    if val_loader:
        evaluator.run(val_loader)
        log_results("val", evaluator.state.metrics, trainer.state.epoch)

    # Test data.
    if test_loader:
        evaluator.run(test_loader)
        log_results("test", evaluator.state.metrics, trainer.state.epoch)

    print()
    print("-" * 80)
    print()

# Start training.
trainer.run(train_loader, max_epochs=num_epochs)
