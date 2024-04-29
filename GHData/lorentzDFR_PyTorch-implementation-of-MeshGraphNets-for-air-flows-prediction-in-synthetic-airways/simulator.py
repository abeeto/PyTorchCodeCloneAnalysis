import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.data import Batch

from Graph_network_functions import EncodeProcessDecode
from Utility_functions import Normalizer as Normalizer

import time 

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# WITHOUT DYNAMIC NORMALIZATION

class Simulator(nn.Module): 

  def __init__(self, connectivity_radius: float, input_size: int, latent_size: int, mlp_num_hidden_layers: int, 
               num_message_passing_steps: int, output_size: int):

    super().__init__()
    self._radius = connectivity_radius
    self.graph_network = EncodeProcessDecode(input_size, latent_size, mlp_num_hidden_layers, 
                                             num_message_passing_steps, output_size)


  def _encoder_preprocessor(self, graph):
    
    node_feat = graph.x
    
    """ Producing connectivity """
    edge_index = radius_graph(graph.pos, self._radius, graph.batch)

    edge_relativePos = (torch.index_select(graph.pos, 0, edge_index[1]) - torch.index_select(graph.pos, 0, edge_index[0])) / self._radius
    edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True)
    edge_feat = torch.cat([edge_relativePos, edge_relativeDist], dim = -1)

    """ Returning new values for batch of graphs """
    return Batch(x = node_feat, edge_index = edge_index, edge_attr = edge_feat, batch = graph.batch)

  def _decoder_postprocessor(self, prediction, graph_features):

    new_pressure = prediction[:,0]    
    accel = prediction[:,1:]
    old_vel = graph_features[:,1:]
    
    # Euler integrator
    new_vel = old_vel + accel # * dt = 1

    return torch.cat([new_pressure.unsqueeze(1), new_vel], dim= -1)

  def forward(self, graph):

    input_graph = self._encoder_preprocessor(graph)  
    prediction = self.graph_network(input_graph)
    predicted_features = self._decoder_postprocessor(prediction.x, graph.x)

    return Batch(x = predicted_features, edge_index = graph.edge_index, edge_attr = graph.edge_attr, batch = graph.batch)