def MyPlot(train_losses, val_losses, train_accuracy,val_accuracy):
    import matplotlib.pyplot as plt

    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'bo', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.figure()

    plt.show()

    plt.plot(epochs, train_accuracy, 'bo', label='Training Acc')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.figure()

    plt.show()