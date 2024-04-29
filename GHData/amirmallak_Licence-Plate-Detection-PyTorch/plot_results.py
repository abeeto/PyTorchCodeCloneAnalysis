import matplotlib.pyplot as plt

from cv2 import cv2


def plot(model, WIDTH, HEIGHT, train_loader, train_dataset, training_accuracies, training_losses, validation_accuracies,
         validation_losses, test_dataset, test_loader, test_batch_loader):
    # -- Plot Neural Network Loss --
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # -- Plot Neural Network Accuracy --
    plt.plot(training_accuracies)
    plt.plot(validation_accuracies)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Showing our Neural Model predictions on the Test dataset - image wise
    indices_index = 0
    plt.figure(figsize=(40, 80))
    plt.title('Model Prediction Results - Test Dataset (Image wise)')
    for i, (image, _) in zip(range(len(test_dataset)), test_loader):
        plt.subplot(3, 4, i + 1)
        plt.axis('off')

        y_hat = model(image)

        xb, yb = int(y_hat[0][0] * WIDTH), int(y_hat[0][1] * WIDTH)
        xt, yt = int(y_hat[0][2] * WIDTH), int(y_hat[0][3] * WIDTH)

        image_index = test_dataset.indices[indices_index]
        image_path = fr'./Dataset/Cars License Plates/Car_License_Image_{image_index}.jpeg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
        image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 0, 255), 1)
        plt.imshow(image)

        indices_index += 1

    plt.show()

    # ------------------------ Train Batch Results ----------------------

    indices_index = 0
    plt.figure(figsize=(40, 80))
    plt.title('Model Prediction Results - Train Dataset (Batch wise)')
    for image_batch, _ in train_loader:
        if indices_index:
            break
        y_hat_batch = model(image_batch)
        for i in range(len(image_batch)):
            plt.subplot(8, 4, i + 1)
            plt.axis('off')

            y_hat = y_hat_batch[indices_index]
            xb, yb = int(y_hat[0].detach().numpy().max() * WIDTH), int(y_hat[1].detach().numpy().max() * WIDTH)
            xt, yt = int(y_hat[2].detach().numpy().max() * WIDTH), int(y_hat[3].detach().numpy().max() * WIDTH)

            image_index = train_dataset.indices[indices_index]
            image_path = fr'./Dataset/Cars License Plates/Car_License_Image_{image_index}.jpeg'
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
            image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 0, 255), 1)
            plt.imshow(image)

            indices_index += 1

    plt.show()

    # ------------------------- Test Batch Results -----------------------

    indices_index = 0
    plt.figure(figsize=(40, 80))
    plt.title('Model Prediction Results - Test Dataset (Batch wise)')
    for image_batch, _ in test_batch_loader:
        y_hat_batch = model(image_batch)
        for i in range(len(test_dataset)):
            plt.subplot(3, 4, i + 1)
            plt.axis('off')

            y_hat = y_hat_batch[indices_index]
            xb, yb = int(y_hat[0].detach().numpy().max() * WIDTH), int(y_hat[1].detach().numpy().max() * WIDTH)
            xt, yt = int(y_hat[2].detach().numpy().max() * WIDTH), int(y_hat[3].detach().numpy().max() * WIDTH)

            image_index = test_dataset.indices[indices_index]
            image_path = fr'./Dataset/Cars License Plates/Car_License_Image_{image_index}.jpeg'
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
            image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 0, 255), 1)
            plt.imshow(image)

            indices_index += 1

    plt.show()
