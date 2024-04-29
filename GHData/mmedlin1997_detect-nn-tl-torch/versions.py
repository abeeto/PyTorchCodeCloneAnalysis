import torch
import torchvision
import matplotlib
import sklearn
import cv2
import PIL
from platform import python_version

def show_versions():
  print("Versions...")
  print("python", python_version())
  print("torch", torch.__version__)
  print("torchvision", torchvision.__version__)
  print("matplotlib", matplotlib.__version__)
  print("sklearn", sklearn.__version__)
  print("cv2", cv2.__version__)
  print("PIL", PIL.__version__)

show_versions()
