from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import DataLoader

################################ Load dataset ####################################
# Benchmark dataset for semi-supervised graph node classification
planetoid = Planetoid(root='/media/lepoeme20/Data/graphs/', name='Cora')

# print dataset volumn
print(len(planetoid))

# print dataset unique label number
print(planetoid.num_classes)

# print number of node features
print(planetoid.num_node_features)

data = planetoid[0]

# following results denote against which number of  node to each phase
print(data.train_mask.sum().item())
print(data.val_mask.sum().item())
print(data.test_mask.sum().item())


############################### Mini-batches ###################################
dataset = TUDataset(root='/media/lepoeme20/Data/graphs/', name='ENZYMES', use_node_attr=True)

# print dataset volumn
print(len(dataset))

# print dataset unique label number
print(dataset.num_classes)

# print number of node features
print(dataset.num_node_labels)

# Access the graph in the dataset
data = dataset[0]

# There are 168/2 = 84 undirected edges
# There are 37 nodes with 3 features
# There is 1 class
print(data)

# split dataset to train_dataset and test_dataset
train_dataset = dataset[:540]
test_dataset = dataset[540:]

# random permutation
dataset = dataset.shuffle

# set dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)
    print(data.num_graphs)
    print(data.num_node_features)

    # Averages data.x values with the same batch index
    x = scatter_mean(data.x, data.batch, dim=0)
