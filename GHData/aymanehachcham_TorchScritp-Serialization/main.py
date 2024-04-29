
import torch
import torchvision
from Modules.deeplab_implementation import SemanticSeg
from Modules.dataset import SegmentationSample

# An instance of my Semantic Segmentation model:
semantic_model = SemanticSeg(pretrained=True, device='cuda')

# An example of input normally provided to the model:
path = './'
filename = 'image-test.jpeg'
image = SegmentationSample(root_dir=path, image_file=filename, device='cuda')



# Call to the forward method fo the model:
#output = semantic_model(image)


# Generate a torchscrpt module via tracing:
# traced_script_module = torch.jit.trace(semantic_model, image)
#
# output = traced_script_module(image.processed_image)


