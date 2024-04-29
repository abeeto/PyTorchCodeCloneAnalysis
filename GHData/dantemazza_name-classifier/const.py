import torch
import numpy as np
import configuration as config
name_map = {}
M_names = []
F_names = []

last_letters = []
di_suffix = []
tri_suffix = []

bigrams = []
trigrams = []

training_set = {}
cv_set = {}
test_set = {}

featureCount = 0

X_train = np.zeros(shape=(featureCount, config.training_size)).transpose
y_train = np.zeros(config.training_size)

X_cv = np.zeros(shape=(featureCount, config.cv_set_size))
y_cv = np.zeros(config.cv_set_size)

X_test = np.zeros(shape=(featureCount, config.test_set_size))
y_test = np.zeros(config.test_set_size)

featureList = []




