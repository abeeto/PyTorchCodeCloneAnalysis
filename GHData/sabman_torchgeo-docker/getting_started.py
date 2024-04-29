# Imports
# Next, we import TorchGeo and any other libraries we need.

import os
import tempfile

from torch.utils.data import DataLoader

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

# Datasets
# ========
# For this tutorial, we’ll be using imagery from the National Agriculture Imagery Program (NAIP) and labels from the Chesapeake Bay High-Resolution Land Cover Project. First, we manually download a few NAIP tiles and create a PyTorch Dataset.

data_root = tempfile.gettempdir()
naip_root = os.path.join(data_root, "naip")
naip_url = "https://naipblobs.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
tiles = [
    "m_3807511_ne_18_060_20181104.tif",
    "m_3807511_se_18_060_20181104.tif",
    "m_3807512_nw_18_060_20180815.tif",
    "m_3807512_sw_18_060_20180815.tif",
]
for tile in tiles:
    download_url(naip_url + tile, naip_root)

naip = NAIP(naip_root)

# Next, we tell TorchGeo to automatically download the corresponding Chesapeake labels.
chesapeake_root = os.path.join(data_root, "chesapeake")
chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)

# Finally, we create an IntersectionDataset so that we can automatically sample from both GeoDatasets simultaneously.
dataset = naip & chesapeake

# Sampler
# =======
# Unlike typical PyTorch Datasets, TorchGeo GeoDatasets are indexed using lat/long/time bounding boxes. This requires us to use a custom GeoSampler instead of the default sampler/batch_sampler that comes with PyTorch.
sampler = RandomGeoSampler(naip, size=1000, length=10)

# DataLoader
# ==========
# Now that we have a Dataset and Sampler, we can combine these into a single DataLoader.
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

# Other than that, the rest of the training pipeline is the same as it is for torchvision.
for sample in dataloader:
    image = sample["image"]
    target = sample["mask"]
