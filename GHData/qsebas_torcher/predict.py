#!/usr/bin/env python

import os
import numpy as np
import pickle


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# max number of words in each sentence
SEQUENCE_LENGTH = 300
# number of words to use, discarding the rest
N_WORDS = 10000
# out of vocabulary token
OOV_TOKEN = None
# 30% testing set, 70% training set
TEST_SIZE = 0.3
# number of CELL layers

model = load_model(os.path.join ("results", "imdb-LSTM-seq-300-em-300-w-10000-layers-1-units-128-opt-adam-BS-64-d-0.4.h5"))

# loading
with open(os.path.join ("results", 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequences
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    return prediction, {0: "negative", 1: "positive"}[np.argmax(prediction)]

print("\n")
print("\n")

text = "What ensues is a film that isn't interested in planting seeds for future Marvel movies -- which is actually a refreshing change of pace."
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real: positive - Prediction:", prediction)
print("\n")

text = "Almost completely lacking in Black Widow is any sense of outrage over the grotesque violations of human rights and human dignity at the root of the whole premise of the film."
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real: negative - Prediction:", prediction)
print("\n")


text = "While humour is an important part of the film, it's used to serve the tragic story rather than undercut it."
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real: positive - Prediction:", prediction)
print("\n")


text = "Black Widow represents the MCU looking back when it should be moving forward. Everything about the movie seems small, even the big action set-pieces"
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real negative - Prediction:", prediction)
print("\n")

text = "More women is its only decent idea. Otherwise, it's business as usual. Otherwise, it's all formulaic bish-bosh, smash-crash action scenes broken up by lame jokes and lame philosophising along the lines of: 'Your pain only makes you stronger.'"
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real negative - Prediction:", prediction)
print("\n")

text = "Shortland does a solid job with the material, but even she isn't able to entirely gloss over how Natasha randomly keeps disappearing in her own movie or how what is supposed to be her story frequently focuses on other characters."
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real posituve - Prediction:", prediction)
print("\n")


text = "BLACK WIDOW is Marvels take on a somewhat silly, over-the-top retro Bond movie. Certainly not great cinematic art, but really entertaining and with a wonderful cast."
print(text)
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("real positive - Prediction:", prediction)
print("\n")
