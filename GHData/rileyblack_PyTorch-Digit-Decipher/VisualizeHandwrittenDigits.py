# ----------------------------------------------------------------------------------------------------------------------
# NOTE: excessive in-line comments were just for my own learning purposes
# ----------------------------------------------------------------------------------------------------------------------

# Importing necessary libraries
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt                                                             # For plotting
from sklearn.model_selection import train_test_split                                        # For easy data splitting

# Importing 28x28 grayscale images from 10 classes of hand written digits (60k for training, 10k for validation/testing)
(x_training, y_training), (x_testing, y_testing) = mnist.load_data()

# Evenly splitting testing data into validation set and test set (validation set used to intermediately evaluate "test" performance throughout learning)
x_validation, x_testing, y_validation, y_testing = train_test_split(x_testing, y_testing, test_size=0.5, random_state=1)

# Defining useful constants
image_dimensions = x_training.shape[1:]
number_of_labels = int(y_training.max() - y_training.min() + 1)
number_of_samples = 5
epochs = 10

# Printing dataset characteristics
print("\n\nDATASET CHARACTERISTICS")
print("-------------------------------------------------------")
print(f"Dimensions of each image: {image_dimensions}")
print(f"Number of Classes: {number_of_labels}")
print(f"Dimensions of training inputs: {x_training.shape}")
print(f"Dimensions of training labels: {y_training.shape}")
print(f"Dimensions of validation inputs: {x_validation.shape}")
print(f"Dimensions of validation labels: {y_validation.shape}")
print(f"Dimensions of testing inputs: {x_testing.shape}")
print(f"Dimensions of testing labels: {y_testing.shape}")

# Displaying 5 samples of each label (class) 0 through 9
plt.figure(figsize=(number_of_labels, number_of_samples*10))                                # Defining empty plot with adequate space
index = 1                                                                                   # Defining next open subplot location
for label in range(number_of_labels):                                                       # For each label (class) 0 through 9
    samples = np.random.choice(np.argwhere(y_training == label)[:, 0], number_of_samples)   # Selecting x data from 5 samples with that label
    for j in range(number_of_samples):                                                      # For each selected sample
        plt.subplot(number_of_labels, number_of_samples, index)                             # Creating subplot in next open location
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_training[samples[j]], cmap=plt.cm.binary)                              # Plotting sample
        index += 1                                                                          # Updating next open subplot location
plt.show()                                                                                  # Displaying result
