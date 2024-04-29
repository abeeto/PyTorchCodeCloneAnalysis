# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import util

char2id = util.get_char2id()

def generate_number():
  number = [random.choice(list('012345678')) for _ in range(random.randint(1, 3)) ]
  return int(''.join(number))

def add_padding(number, is_input=True):
  number = "{: <7}".format(number) if is_input else "{: <5s}".format(number)
  return number

def prepare_data():
  # Prepare data
  input_data = []
  output_data = []

  # Prepare 50,000 data
  while len(input_data) < 50000:
    x = generate_number()
    y = generate_number()
    z = x - y
    input_char = add_padding(str(x) + '-' + str(y))
    output_char = add_padding('_' + str(z), is_input=False)

    input_data.append([char2id[c] for c in input_char])
    output_data.append([char2id[c] for c in output_char])

  return input_data, output_data

# Devide into 7:3 data
#train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size=0.7)

# Convert data into batch
def train2batch(input_data, output_data, batch_size=100):
  input_batch = []
  output_batch = []
  input_shuffle, output_shuffle = shuffle(input_data, output_data)
  for i in range(0, len(input_data), batch_size):
    input_batch.append(input_shuffle[i:i+batch_size])
    output_batch.append(output_shuffle[i:i+batch_size])
  return input_batch, output_batch

#i, o = train2batch(train_x, train_y)
#breakpoint()
