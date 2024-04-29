import os
cuda_num = 0

#fzh changed if main_fer2013.py class_number = 6 elif main_omg.py class_number = 7
#fzh changed if main_fer2013.py model_path = './models_fer2013/'  elif main_omg.py model_path = './models_omg/'
# mode changed
mode = 'fer2013'

if mode=='fer2013':
    class_number = 7
    model_path = './models_fer2013_deep_short_dlib/'
elif mode=='omg':
    class_number = 7
    model_path = './models_omg/'

lr = 1e-4
epoch = 120
batch_size = 1
thresh=0.5

data_root = 'data/'

#fzh changed
# image_dir = os.path.join(data_root, 'face_images/')
image_dir = '/home/fzh/AI/face_emotion/DRML_pytorch/data/img'
csv_dir='/home/fzh/AI/face_emotion/DRML_pytorch/data/dev.csv'

