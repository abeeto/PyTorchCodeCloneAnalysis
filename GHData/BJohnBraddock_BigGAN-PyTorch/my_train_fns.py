import torch
import torch.nn.functional as F
import my_losses


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train

def VCA_latent_training_function(G, VCA, z_, y_, z_y_optim, config):
  
  def train():
    z_y_optim.zero_grad()

    G_z = G(z_, y_)

    G_z = F.interpolate(G_z, size=224)
    
    VCA_G_z = VCA(G_z).view(-1)

    if config['train_unpleasant']:
      G_loss = my_losses.generator_vca_loss_unpleasant(VCA_G_z)
    else:
      G_loss = my_losses.generator_vca_loss_pleasant(VCA_G_z)
    
    G_loss.backward()
    z_y_optim.step()

    out = {
      'G_loss': float(G_loss.item()),
      'VCA_G_z': torch.mean(VCA_G_z).item()
    }

    return out
  
  return train

def VCA_latent_training_function_alt(G, VCA, z_, y_, z_y_optim, config):
  
  def train():
    z_y_optim.zero_grad()

    G_z = G(z_, y_, truncation=config['truncation'])

    G_z = F.interpolate(G_z, size=224)
    
    VCA_G_z = VCA(G_z).view(-1)

    if config['train_unpleasant']:
      G_loss = my_losses.generator_vca_loss_unpleasant(VCA_G_z)
    else:
      G_loss = my_losses.generator_vca_loss_pleasant(VCA_G_z)
    
    G_loss.backward()
    z_y_optim.step()

    out = {
      'G_loss': float(G_loss.item()),
      'VCA_G_z': torch.mean(VCA_G_z).item()
    }

    return out
  
  return train

def alexnet_latent_training_function(G, alexnet, z_, y_, z_y_optim, config):
  
  def train():
    z_y_optim.zero_grad()

    G_z = G(z_, y_)

    G_z = F.interpolate(G_z, size=224)
    
    alexnet_G_z = F.softmax(alexnet(G_z)[0], dim=0)

    alexnet_G_z = alexnet_G_z[config['alexnet_class']]

    G_loss = my_losses.generator_vca_loss_pleasant(alexnet_G_z)
    
    G_loss.backward()
    z_y_optim.step()

    out = {
      'G_loss': float(G_loss.item()),
      'alexnet_G_z': torch.mean(alexnet_G_z).item()
    }

    return out
  
  return train


def VCA_generator_training_function(G, VCA, z_, y_, config):


  def train():
    G.optim.zero_grad()

    z_.sample_()
    y_.sample_()

    G_z = G(z_[:config['batch_size']], G.shared(y_[:config['batch_size']]))

    
    # Resize image
    G_z = F.interpolate(G_z, size=224)

    VCA_G_z = VCA(G_z).view(-1)
    #TODO: Should this loss be reversed?....

    if config['train_unpleasant']:
      G_loss = my_losses.generator_vca_loss_unpleasant(VCA_G_z)
    else:
      G_loss = my_losses.generator_vca_loss_pleasant(VCA_G_z)
      
    G_loss.backward()

    G.optim.step()

    out = {'G_loss': float(G_loss.item()), 'VCA_G_z': torch.mean(VCA_G_z).item()}
    return out

  return train