import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm, trange
import os
import plotly.graph_objects as go



def print_3D_graph(nodes, edges, color = "royalblue"):
  """ Function used to plot a point cloud with connectivity (graph) """

  trace1 = go.Scatter3d(
      x=nodes[:,0],
      y=nodes[:,1],
      z=nodes[:,2],
      mode = 'markers',
      showlegend=False,
      marker = dict(size=2, color = color, colorscale = 'turbo', showscale=True)
      )
  
  x_lines = []
  y_lines = []
  z_lines = []

  for i in range(edges.shape[0]):
      x_lines.append(nodes[edges[i,0],0])
      y_lines.append(nodes[edges[i,0],1])
      z_lines.append(nodes[edges[i,0],2])
      
      x_lines.append(nodes[edges[i,1],0])
      y_lines.append(nodes[edges[i,1],1])
      z_lines.append(nodes[edges[i,1],2])
      x_lines.append(None)
      y_lines.append(None)
      z_lines.append(None)


  trace2 = go.Scatter3d(
      x = x_lines,
      y = y_lines,
      z = z_lines,
      mode = 'lines',
      name = 'edge',
      showlegend=False,
      line = dict(width=1, color="grey"),
      opacity=0.5
      )

  layout = go.Layout(scene = dict(aspectmode='data'))  #aspectratio=dict(x=1, y=1, z=1)))
  
  fig = go.Figure(data=[trace1, trace2], layout = layout)
  fig.update_scenes(zaxis_autorange="reversed")
  fig.show()





class Graph_dataset(Dataset):

  def __init__(self, root, window=1, transform=None, pre_transform=None, pre_filter=None):

    """ Root: where the dataset should be stored """
    super().__init__(root, transform, pre_transform, pre_filter)
    self._window = window
    
  @property
  def raw_file_names(self):
    return 'Dataset.pt'

  @property
  def processed_file_names(self):

    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))

    return [f'data_{i}.pt' for i in trange(self.data.shape[0])]

  def download(self):
    pass


  def process(self):

    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))
    
    for i, graph in enumerate(tqdm(self.data)):
    
      node_pos = graph[..., :3].detach().clone()
      node_features = graph[..., 3:].detach().clone()
      data = Data(x = node_features, pos = node_pos)
      torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))


  def len(self):        
    return self.data.shape[0]# - self._window


  def get(self, idx):
    """ Equivalent to __getitem__ in pytorch """ 

    sample = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    target = torch.load(os.path.join(self.processed_dir, f'data_{idx+self._window}.pt'))

    return sample, target