import torch
import torch.nn as nn


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim=64):
        super(DeterministicEncoder, self).__init__()
        self.build_default_network(x_dim + y_dim, r_dim)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor):
        data_context = torch.cat(x_context, y_context)
        r_i = self.network(data_context)
        r_c = torch.mean(r_i, dim=0)
        return r_c

    def build_default_network(self, in_dim, out_dim):
        layer_count = 1
        h_dim = 128
        self.network = nn.Sequential()
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(in_dim, h_dim))
        self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
        layer_count += 1
        for _ in range(4):
            self.network.add_module("linear_{}".format(layer_count), nn.Linear(h_dim, h_dim))
            self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
            layer_count += 1
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(h_dim, out_dim))


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim=64):
        super(LatentEncoder, self).__init__()
        self.build_default_network(x_dim + y_dim, z_dim)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor):
        data_context = torch.cat(x_context, y_context)
        s_i = self.network(data_context)
        s_c = torch.mean(s_i, dim=0)
        s_c = self.aggregate_layer(s_c)
        z_loc = self.loc(s_c)
        z_scale = 0.1 + 0.9 * nn.functional.sigmoid(self.scale(s_c))
        return z_loc, z_scale
    
    def build_default_network(self, in_dim, out_dim):
        layer_count = 1
        h_dim = 128
        self.network = nn.Sequential()
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(in_dim, h_dim))
        self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
        layer_count += 1
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(h_dim, h_dim))
        self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
        layer_count += 1
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(h_dim, out_dim))
        self.aggregate_layer = nn.Sequential(
            nn.Linear(h_dim, 96),
            nn.ReLU()
        )
        self.loc = nn.Linear(96, out_dim)
        self.scale = nn.Linear(96, out_dim)


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, r_dim, y_dim):
        super(Decoder, self).__init__()
        self.build_default_network(x_dim + r_dim + z_dim, y_dim)
    
    def forward(self, x_target, r, z):
        num_target_points = x_target.shape[0]
        data = torch.cat((x_target.reshape(-1,1), r.repeat(num_target_points, 1), z.repeat(num_target_points, 1)), dim=1)
        hidden = self.network(data)
        loc = self.loc(hidden)
        scale = 0.1 + 0.9 * nn.functional.softplus(self.scale(hidden))
        return loc, scale

    def build_default_network(self, in_dim, out_dim):
        self.network = nn.Sequential()
        layer_count = 1
        h_dim = 128
        self.network = nn.Sequential()
        self.network.add_module("linear_{}".format(layer_count), nn.Linear(in_dim, h_dim))
        self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
        layer_count += 1
        for _ in range(2):
            self.network.add_module("linear_{}".format(layer_count), nn.Linear(h_dim, h_dim))
            self.network.add_module("relu_{}".format(layer_count), nn.ReLU())
            layer_count += 1
        self.loc = nn.Linear(h_dim, out_dim)
        self.scale = nn.Linear(h_dim, out_dim)
