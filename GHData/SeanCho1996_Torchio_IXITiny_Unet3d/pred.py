# %% import dependencies
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import torch.nn as nn
import torch

import pickle

import torchio as tio

from model import UNet

# %% some global variables 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"predictions are made on device: {device}")

# %% load model params and preprocess params
model_param_name = f"./Unet3d_IXITiny.pth"
state_dict = torch.load(model_param_name, map_location=device)

landmarks_name = "./landmarks.pkl"
with open(landmarks_name, 'rb') as handle:
   saved_data = pickle.load(handle)
landmarks = saved_data['landmarks']
num_classes = saved_data['num_classes']

# %% build model from state_dict
model = UNet(num_classes)
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model = nn.DataParallel(model)
model.to(device)
model.eval()

# %% predict data process
img_file = os.path.join(os.path.dirname(__file__), "ixi_tiny", "image", "IXI014-HH-1236_image.nii.gz")
subject_embedding = tio.Subject(
                mri=tio.ScalarImage(img_file)
                )

validation_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((48, 60, 48)),
        tio.HistogramStandardization({'mri': landmarks}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.OneHot(),
    ])

test_data = validation_transform(subject_embedding)

patch_size = 48, 48, 48  # we can user larger patches for inference
patch_overlap = 4, 4, 4
grid_sampler = tio.inference.GridSampler(
    test_data,
    patch_size,
    patch_overlap,
)
patch_loader = torch.utils.data.DataLoader(
    grid_sampler, batch_size=16)
aggregator = tio.inference.GridAggregator(grid_sampler)

# %% make prediction
with torch.no_grad():
    for patches_batch in patch_loader:
        inputs = patches_batch['mri'][tio.DATA].to(device)
        locations = patches_batch[tio.LOCATION]
        probabilities = model(inputs).softmax(dim=1)
        aggregator.add_batch(probabilities, locations)

output_tensor = aggregator.get_output_tensor()  # this is already the output tensor, but it's only a tensor, not a file-like object
output_tensor = torch.argmax(output_tensor, dim=0).unsqueeze(0)

affine = test_data.mri.affine
prediction = tio.LabelMap(tensor=output_tensor.short(), affine=affine)

spatial_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4)
])
resample_shape = spatial_transform(tio.ScalarImage(img_file)).shape[-3:]

prediction = tio.CropOrPad(resample_shape)(prediction)
prediction = tio.Resample(img_file, image_interpolation='nearest')(prediction)

save_name = f'res.nii.gz'
prediction.save(save_name)