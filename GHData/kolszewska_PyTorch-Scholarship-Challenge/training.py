from torch import nn, optim

import data_helper
import model_helper
import training_helper

num_epochs = 200
batch_size = 32
dropout_probability = 0.5
number_of_classes = 102
learning_rate = 0.001
scheduler_step_size = 5
scheduler_gamma = 0.1

train_directory = 'flower_data/train'
validation_directory = 'flower_data/valid'

train_loader, validation_loader, total_training_batches = data_helper.load_data(train_directory,
                                                                                validation_directory,
                                                                                batch_size)

model = model_helper.get_model_for_training()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

training_helper.train(train_loader, validation_loader, num_epochs, total_training_batches,
                      model, criterion, optimizer, scheduler)
