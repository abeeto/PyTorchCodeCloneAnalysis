"""Hyperparameters to be used by the application"""

hps = {
    'learning_rate': .01, # for the Adam optimizer
    'dropout': .1,
    'lstm_layers': 3, # how many LSTM modules to stack in the network
    'batch_size': 64, # how many names to train on at once
    'epochs': 5000000000, # number of training epochs
    'print_every': 1, # print the loss every n epochs
    'onehot_length': 27, # number of categories (all lowercase letters and _)
    'save_every': 1, # how often to save the model
    'log_every': 1, # how often to log the loss
}
