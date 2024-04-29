import torch
import wandb
import multiprocessing
from torchinfo import summary
from alive_progress import alive_bar

from models.UNet import UNet
from models.ResUNet import ResUNet
from models.ConvNeXtUNet import ConvNeXtUNet

from dataloader import A2D2_Dataset, CLASS_LABELS
from utils import ArgmaxIOU, convertPredictionsForWB, save_model

if __name__ == "__main__":

    # Setting up the system for cude if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Constant
    BATCH_SIZE = 3
    IMG_SIZE = (400, 400)  # 424 ?
    LEARNING_RATE = 0.01
    WORKERS = multiprocessing.cpu_count()
    FEATURES = 16
    KERNEL_SIZE = 5

    MAX_EPOCHS = 250

    # Open datasets
    train_dataset = A2D2_Dataset("training", size=IMG_SIZE)
    # test_dataset = A2D2_Dataset("testing", size=IMG_SIZE)
    val_dataset = A2D2_Dataset("validation", size=IMG_SIZE)

    # Define data loader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=WORKERS)
    # data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Create model
    model = ResUNet(3, train_dataset.n_class(), features=FEATURES, kernel_size=KERNEL_SIZE)
    model.to(device)

    # Summary
    summary(model, input_size=(BATCH_SIZE, 3) + IMG_SIZE)

    # Init W&B
    config = {
        "model_name": model._get_name(),
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "learning_rate": LEARNING_RATE,
        "workers": WORKERS,
        "features": FEATURES,
        "kernel_size": KERNEL_SIZE
    }

    wandb.init(project="Road Segmentation PyTorch", config=config)

    ##################################################################################################################################
    # Using presaved model
    ##################################################################################################################################
    # model.load_state_dict(torch.load("./ResUNet_size-512-512_features-16_epoch-7.pth", map_location=device))
    ##################################################################################################################################

    # Watch model for W&B
    wandb.watch(model, log_freq=1000)

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Metrics
    metric_argmax = ArgmaxIOU(train_dataset.n_class())

    # Loop over the dataset multiple epochs
    for epoch in range(MAX_EPOCHS):

        ########################
        #       TRAINING       #
        ########################

        # Set the model for training
        model.train()

        training_loss = 0.0
        training_argmax = 0.0

        with alive_bar(len(data_loader_train), length=10, title="Epoch : " + str(epoch) + " - Training") as bar:

            # Iterate trough the batches of the dataloader
            for i, data in enumerate(data_loader_train, 1):

                # get the inputs; data is a list of [inputs, targets] to the correct device
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Clear the gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward Pass
                outputs = model(inputs)

                # Find the Loss
                loss = criterion(outputs, targets)

                # Calculate gradients
                loss.backward()

                # Update Weights
                optimizer.step()

                # Calculate Loss
                training_loss += loss.item()

                # Calculate metrics
                training_argmax += metric_argmax(outputs, targets)

                # Update the progress bar
                bar.text(f'-> Loss: {training_loss / i:.3f} - Argmax: {training_argmax / i:.3f}')
                bar()

        ########################
        #      VALIDATION      #
        ########################

        # Set the model for infering
        model.eval()

        validation_loss = 0.0
        validation_argmax = 0.0

        image_log_wb = []

        with alive_bar(len(data_loader_val), length=10, title="Epoch : " + str(epoch) + " - Validation") as bar:

            # Disabling the gradient
            with torch.no_grad():

                # Iterate trough the batches of the dataloader
                for i, data in enumerate(data_loader_val, 1):

                    # get the inputs; data is a list of [inputs, targets] to the correct device
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward Pass
                    outputs = model(inputs)

                    # Find the Loss
                    val_loss = criterion(outputs, targets)

                    # Calculate Loss
                    validation_loss += val_loss.item()

                    # Calculate metrics
                    validation_argmax += metric_argmax(outputs, targets)

                    if i < 25:
                        image_log_wb.append((inputs[0], outputs[0], targets[0]))

                    # Update the progress bar
                    bar.text(f'-> Loss: {validation_loss / i:.3f} - Argmax: {validation_argmax/i:.3f}')
                    bar()

        ########################
        #        LOGGING       #
        ########################

        # Average of loss
        training_loss = training_loss / len(data_loader_train)
        validation_loss = validation_loss / len(data_loader_val)
        training_argmax = training_argmax / len(data_loader_train)
        validation_argmax = validation_argmax / len(data_loader_val)

        # Logging
        print(f"Epoch : {epoch} - Train loss {training_loss:.3f} - Val loss {validation_loss:.3f} - Train Argmax: {training_argmax:.3f} - Val Argmax: {validation_argmax:.3f}\n")
        wandb.log({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_argmax": training_argmax,
            "validation_argmax": validation_argmax,
            "predictions": convertPredictionsForWB(image_log_wb, CLASS_LABELS)
        })

        # Save the model
        # torch.save(model.state_dict(), f'./{model._get_name()}_epoch-{epoch}_v-argmax{validation_argmax:.3f}.pth')
        save_model(f'./{model._get_name()}_size-{IMG_SIZE[0]}-{IMG_SIZE[1]}_epoch-{epoch}_v-argmax-{validation_argmax:.3f}.pth',
                   model,
                   optimizer,
                   epoch,
                   config,
                   A2D2_Dataset.static_data())