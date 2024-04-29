import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
# from torch_geometric.nn import LayerNorm


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Custom_mlp(nn.Module):

  def __init__(self, input_size: int, latent_size: int, mlp_num_hidden_layers: int, layernorm = False):

    super().__init__()

    layers = []
    for i in range(mlp_num_hidden_layers):
      
      if i == 0:
        layers += [nn.Sequential(nn.Linear(input_size, latent_size), nn.ReLU())]
      else:      
        layers += [nn.Sequential(nn.Linear(latent_size, latent_size), nn.ReLU())]       
    
    layers += [nn.Linear(latent_size, latent_size)]
    self.network = nn.Sequential(*layers)
    
    if layernorm == True:
        
      self.network = nn.Sequential(self.network, nn.LayerNorm(normalized_shape = latent_size))
#       self.network = nn.Sequential(self.network, LayerNorm(in_channels = latent_size)) # from geometric

                             
  def forward(self, x):
    
    out = self.network(x)
    return out





class GNN(MessagePassing):

  def __init__(self, latent_size: int, mlp_num_hidden_layers: int):
    super().__init__(aggr='add')

    self.message_func = Custom_mlp(3*latent_size, latent_size, mlp_num_hidden_layers, layernorm = True)
    self.update_func = Custom_mlp(2*latent_size, latent_size, mlp_num_hidden_layers, layernorm = True)

  def forward(self, graph):
    # features has shape [N, in_features]
    # edge_attr has shape [E, in_features]
    # edge_index has shape [2, E]

    features = graph.x
    edge_attr = graph.edge_attr
    edge_index = graph.edge_index
    new_edge_attr = self.edge_updater(edge_index, x = features, edge_attr = edge_attr)

    return self.propagate(edge_index, x = features, edge_attr = new_edge_attr, batch = graph.batch)


  def edge_update(self, edge_index, x_i, x_j, edge_attr):
    # x_i has shape [E, in_features]
    # x_j has shape [E, in_features]
    # edge_attr has shape [E, in_features]        

    """ Concatenate the attributes and pass them to an mlp to obtain new edge attributes """
    new_edge_attr = self.message_func(torch.cat([x_i, x_j, edge_attr], dim = 1))

    return new_edge_attr


  def message(self, x_i, x_j, edge_attr):
    # edge_attr has shape [E, out_features]

    return edge_attr    


  def update(self, aggr_features, x, edge_index, edge_attr, batch):
    # x has shape [N, in_features]
    # new_edge_attr has shape [N, out_features]
    # aggr_features has shape [N, out_features]

    """ Concatenate the attributes and pass them to an mlp to update node features """
    new_features = self.update_func(torch.cat([x, aggr_features], dim = 1))

    return Batch(x = new_features, edge_index = edge_index, edge_attr = edge_attr, batch = batch)





class GNN_block(nn.Module):

  def __init__(self, latent_size: int, mlp_num_hidden_layers: int):

    super().__init__()

    self.gnn = GNN(latent_size, mlp_num_hidden_layers)

  def forward(self, latent_graph):
    """ One step of message passing. """
    new_latent_graph = self.gnn(latent_graph)
    
    """ Residual connections for node and edge features """
    new_latent_graph.x += latent_graph.x
    new_latent_graph.edge_attr += latent_graph.edge_attr

    return new_latent_graph




class Encoder(nn.Module):
  """ Encodes the graph input features into latent features """

  def __init__(self, input_size: int, latent_size: int, mlp_num_hidden_layers: int):
    
    super().__init__()

    self._input_size = input_size
    self._latent_size = latent_size
    self.node_model = Custom_mlp(self._input_size, self._latent_size, mlp_num_hidden_layers, layernorm = True)#.to(dev)
    self.edge_model = Custom_mlp(self._input_size, self._latent_size, mlp_num_hidden_layers, layernorm = True)#.to(dev)
    
  def forward(self, graph):
    
    encoded_node_features = self.node_model(graph.x)
    encoded_edge_features = self.edge_model(graph.edge_attr)
    encoded_graph = Batch(x = encoded_node_features, edge_index = graph.edge_index, edge_attr = encoded_edge_features, batch = graph.batch)

    return encoded_graph





class Processor(nn.Module):
  """ Processes the latent graph with several steps of message passing """

  def __init__(self, latent_size: int, mlp_num_hidden_layers: int, num_message_passing_steps: int):
    
    super().__init__()
    self._latent_size = latent_size
    self._num_message_passing_steps = num_message_passing_steps
  
    """ Do `m` message passing steps for the latent graph """
    gnn_blocks = []

    for _ in range(self._num_message_passing_steps):      
        gnn_blocks += [GNN_block(latent_size, mlp_num_hidden_layers)]

    self.network = nn.Sequential(*gnn_blocks)

  def forward(self, encoded_graph):

    processed_graph = self.network(encoded_graph)
    return processed_graph





class Decoder(nn.Module):
  """ Decodes node features from the latent graph """

  def __init__(self, latent_size: int, output_size: int, mlp_num_hidden_layers: int):

    super().__init__()

    self._output_size = output_size
    self.mlp = Custom_mlp(latent_size, output_size, mlp_num_hidden_layers)

  def forward(self, processed_graph):
    """ Decodes the gnn output to obtain the prediction """
    
    decoded_features = self.mlp(processed_graph.x)
    decoded_graph = Batch(x = decoded_features, edge_index = processed_graph.edge_index, 
                          edge_attr = processed_graph.edge_attr, batch = processed_graph.batch)
    return decoded_graph





class EncodeProcessDecode(nn.Module):
  """ Core part of the learnable simulator """

  def __init__(
      
      self,
      input_size: int,      
      latent_size: int,
      #mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int):
    
    super().__init__()

    self._latent_size = latent_size
    #self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size

    self.encoder = Encoder(input_size, latent_size, mlp_num_hidden_layers)
    self.processor = Processor(latent_size, mlp_num_hidden_layers, num_message_passing_steps)
    self.decoder = Decoder(latent_size, output_size, mlp_num_hidden_layers)


  def forward(self, preprocessed_graph):

    encoded_graph = self.encoder(preprocessed_graph)
    processed_graph = self.processor(encoded_graph)
    return self.decoder(processed_graph)