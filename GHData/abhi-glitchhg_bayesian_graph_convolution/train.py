import torch 
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.models import InnerProductDecoder
from .graph_autoencoder import BN_GAE, BGCNEncoder
import torch_geometric.transforms as T

def bn_train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss_dict = model.recon_loss(z, train_pos_edge_index, x,nb_samples=3 )
    kl_loss = loss_dict["total_pw"] - loss_dict["total_qw"] 

    loss = 10 * loss_dict["loss"]  + 1*kl_loss / len(x)  # the multiplying factors are just hyperparameters 
    loss.backward()
    optimizer.step()
    return loss


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

writer=SummaryWriter("./logs/BGAE")
out_channels = 2
num_features = dataset.num_features


model = BN_GAE(encoder=BGCNEncoder(num_features,out_channels),decoder =  InnerProductDecoder())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 100):
    loss = bn_train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    writer.add_scalar("loss",loss,global_step=epoch,new_style=True)
    writer.add_scalar("auc",auc,global_step=epoch,new_style=True)
    writer.add_scalar("ap",ap,global_step=epoch,new_style=True)


