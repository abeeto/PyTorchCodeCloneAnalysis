# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:33:54 2022

@author: Admin
"""

import torch
import torch.nn as nn


class NeRF(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.L_x = 10
        self.L_d = 4
        pos_encoding_c = 3 + 3*2*self.L_x
        dir_encoding_c = 3 + 3*2*self.L_d
        
        in_c = pos_encoding_c
        out_c = 256
        
        #First 5 MLP layers
        num_initial_layers = 5
        initial_layers = []
        
        for i in range(num_initial_layers):
            
            initial_layers.append(nn.Linear(in_c, out_c))
            initial_layers.append(nn.ReLU())
            in_c = out_c
            
        self.initial_layers = nn.Sequential(*initial_layers)
        
        #Next 3 layers
        in_c = pos_encoding_c + out_c
        num_mid_layers = 3
        mid_layers = []
        
        for i in range(num_mid_layers):
            mid_layers.append(nn.Linear(in_c, out_c))
            mid_layers.append(nn.ReLU())
            out_c = in_c
            
        self.mid_layers = nn.Sequential(*mid_layers)
        
        #Sigma layer is used to predict the volume density for 3D points
        #along the rays given the viewing directions.
        
        self.sigma_layer = nn.Linear(out_c, out_c + 1)
        
        self.penultimate_layer = nn.Sequential(nn.Linear(dir_encoding_c + out_c, out_c//2),
                                                nn.ReLU())
        
        self.final_layer = nn.Sequential(nn.Linear(out_c//2, 3),
                                          nn.Sigmoid())
        
    
    def forward(self, spatial_loc, viewing_directions):
        
        #positional encoding for spatial locations (x,y,z)
        spatial_loc_encoded = [spatial_loc]
        
        for l in range(self.L_x):
            
            spatial_loc_encoded.append(torch.sin(2**l * torch.pi * spatial_loc))
            spatial_loc_encoded.append(torch.cos(2**l * torch.pi * spatial_loc))
        
        spatial_loc_encoded = torch.cat(spatial_loc_encoded, dim = -1)
        
        #positional encoding for viewing directions (theta, phi)
        
        viewing_directions = viewing_directions / viewing_directions.norm(p=2, dim = -1).unsqueeze(-1)
        viewing_dirs_encoded = [viewing_directions]
        
        for l in range(self.L_d):
            
            viewing_dirs_encoded.append(torch.sin(2**l * torch.pi * viewing_directions))
            viewing_dirs_encoded.append(torch.cos(2**l * torch.pi * viewing_directions))
            
        viewing_dirs_encoded = torch.cat(viewing_dirs_encoded, dim = -1)
        
        res = self.initial_layers(spatial_loc_encoded)
        res = self.mid_layers(torch.cat([spatial_loc_encoded, res], dim = -1))
        res = self.sigma_layer(res)
        sigma = torch.relu(res[:, 0])
        res = self.penultimate_layer(torch.cat([viewing_dirs_encoded, res[:, 1:]], dim = -1))
        rgb_color = self.final_layer(res)
        
        return {"rgb_color": rgb_color, "sigma": sigma}
    

#model = NeRF()
#print(model)    
        