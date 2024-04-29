"""
@author: Federico Ottomano
"""

import torch
import torch.nn as nn

class BindedVAE(nn.Module):
    
    def __init__(self, mask_input, 
                       mask_latent,
                       ratio_input,
                       ratio_latent):
        
        super().__init__()
        
        self.mask_vae = MaskVAE(mask_input, mask_latent)
        self.ratio_vae = RatioVAE(ratio_input, ratio_latent)
    
    def generate_materials(self, n_samples : int):
        
        noise_mask = torch.randn(n_samples, self.mask_vae.latent_size)
        noise_ratio = torch.randn(n_samples, self.ratio_vae.latent_size)
        
        binaries = self.mask_vae.decoder(noise_mask)
        
        ratio_vae_input = torch.cat([noise_ratio, binaries], dim=1)
        
        output_ratios = self.ratio_vae.decoder(ratio_vae_input)
        
        final_recon = binaries * output_ratios
        
        return final_recon.detach().numpy()
        
    def forward(self, ratios, bins):
        
        (recon_mask, 
         mu_mask, logvar_mask) = self.mask_vae(bins)
        
        (recon_ratio, 
        mu_ratio, logvar_ratio) = self.ratio_vae(ratios, recon_mask)
        
        
        return (recon_mask, mu_mask, logvar_mask,
                    recon_ratio, mu_ratio, logvar_ratio)
        

class MaskVAE(nn.Module):
    
    def __init__(self, input_size, latent_size):
        
        super().__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_size)
            )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
            )
        
    def reparameterize(self, z_mean, z_log_var, batch_size):
        
        epsilon = torch.randn(batch_size, self.latent_size)

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

        
        
    def forward(self, inputs):
                        
        encoding = self.encoder(inputs)
        
        mu, logvar = torch.split(self.encoder(inputs),
                                        split_size_or_sections=self.latent_size,
                                        dim=1)
        
        z = self.reparameterize(mu, logvar, inputs.shape[0])
                
        return self.decoder(z), mu, logvar
    
    
class RatioVAE(nn.Module):
    
    def __init__(self, input_size, latent_size):
        
        super().__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            
            nn.Linear(2 * input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_size),
            )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(self.input_size + self.latent_size , 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
            )
        
    def reparameterize(self, z_mean, z_log_var, batch_size):
        
        epsilon = torch.randn(batch_size, self.latent_size)

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
    def forward(self, ratios, bins):
        
        inputs = torch.cat([ratios, bins], dim=1)
        
        encoding = self.encoder(inputs)
        
        mu, logvar = torch.split(self.encoder(inputs),
                                        split_size_or_sections=self.latent_size,
                                        dim=1)
        
        z = self.reparameterize(mu, logvar, ratios.shape[0])
        
        z = torch.cat([z, bins], dim=1)
                
        return self.decoder(z), mu, logvar
        
        