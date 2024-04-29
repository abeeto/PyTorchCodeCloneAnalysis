# Optional list of dependencies required by the package
dependencies = ["torch", "torchvision"]

# classification
from models.logistic_regression import logistic_regression_model
from models.tissue_tile_net import tissue_tile_net_model, tissue_tile_net_transform, tissue_tile_net_model_5_class

from transforms.TileTissueNet5Class import TissueTileNetTransformer
