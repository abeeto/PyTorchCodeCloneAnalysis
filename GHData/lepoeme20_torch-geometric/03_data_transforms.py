import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/media/lepoeme20/Data/graphs/ShapeNet', categories=['Airplane'])

print(dataset[0])

# Convert the point cloud dataset into a graph dataset
# by generating nearest neighbor graphs from the point clouds via transforms:
transformed_dataset = ShapeNet(root='/media/lepoeme20/Data/graphs/ShapeNet',\
    categories=['Airplane'], pre_transform=T.KNNGraph(k=6), transform=T.RandomTranslate(0.01))

print(transformed_dataset[0])
