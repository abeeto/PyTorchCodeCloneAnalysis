class CONFIG(object):
    """docstring for CONFIG"""

    def __init__(self):
        super(CONFIG, self).__init__()

        self.dataset = 'R8'
        self.learning_rate = 0.02  # Initial learning rate.
        self.epochs = 200  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.  # Dropout rate (1 - keep probability).
        self.early_stopping = 10  # Tolerance for early stopping (# of epochs).
