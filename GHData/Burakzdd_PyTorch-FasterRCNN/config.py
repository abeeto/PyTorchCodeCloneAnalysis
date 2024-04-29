import torch

batch_size = 1
resize_to = 416
epochs = 20
num_workers = 4
device = torch.device('cpu')# if torch.cuda.is_available() else torch.device('cpu')
train_dir = '/home/burakzdd/my_workspace/PyTorch/faster-RCNN/dataset/images/new_train/'
classes = [
    'prohibitory','danger','mandatory','other'
]
visualize_transformed_images = True
out_dir = '/home/burakzdd/my_workspace/PyTorch/faster-RCNN/models/'
model_name = "mask_model"
test_model_name = "model_name"
test_video = "/path_name/"
test_image = ""