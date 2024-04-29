import sys
import moxel
from pretrained_classifier import PretrainedClassifier

model_name = 'SqueezeNet'
classifier = PretrainedClassifier(model_name)


def predict(img):
    img = img.to_PIL()
    label = classifier.predict(img)
    return { 'label': label }
