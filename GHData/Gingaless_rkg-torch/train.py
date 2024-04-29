

import torch
import numpy as np
import sys, os
from torch import optim, nn
from generator import SG2_Generator
from discriminator import SG2_Discriminator
import stylegan1.c_utils as utils
import stylegan1.c_dset as dset
from tqdm import tqdm, trange
import loss
import plr


class __train():

    def __init__(self):
        
        self.__G = None
        self.__D = None
        self.__optim_type = None
        self.__optimG = None
        self.__optimD = None
        self.__lossG = None
        self.__lossD = None
        self.__reg_path_len = True
        self.__total_epochs = 0
        self.__total_D_iter = 0
        self.__total_G_iter = 0
        self.__total_G_losses = []
        self.__total_D_losses = []
        self.__device = None
        self.image_size = 512
        self.batch_size = 8
        self.n_critic = 1
        self.image_path = ''
        self.lazy_reg = 0


    def init(self, G, D, optimizer={'type':'lsgan','args_d':{'lr':0.0}, 'args_g':{'lr':0.0}}, regulate_path_length=True, lazy_reg = 16):
        self.__G = G
        self.__D = D
        self.optimizer = optimizer
        self.__reg_path_len = regulate_path_length
        self.lazy_reg = lazy_reg
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

    def save(self, path):
        s_dict = {'generator' : self.__G.state_dict(), 'discriminator' : self.__D.state_dict()}
        s_dict['optimizer'] = {'type' : self.__optim_type, 'state_dict_optimG' : self.__optimG.state_dict(), 
        'state_dict_optimD' : self.__optimD.state_dict()}
        s_dict['regulate_path_length'] = self.__reg_path_len
        s_dict['epochs'] = self.__total_epochs
        s_dict['iter_of_D'] = self.__total_D_iter
        s_dict['iter_of_G'] = self.__total_G_iter
        s_dict['loss_of_G'] = self.__total_G_losses
        s_dict['loss_of_D'] = self.__total_D_losses
        s_dict['n_critic'] = self.n_critic
        s_dict['device'] = 'gpu' if self.__device.type == 'cuda' else 'cpu'
        s_dict['lazy_regularization'] = self.lazy_reg
        torch.save(s_dict, path)

    def load_from_state_dict(self, s_dict):
        utils.save_device = s_dict['device']
        utils.load_device = 'gpu' if self.__device.type == 'cuda' else 'cpu'
        G = utils.load_model(SG2_Discriminator, s_dict['generator'])
        D = utils.load_model(SG2_Discriminator, s_dict['discriminator'])
        self.init(G, D, {'type' : s_dict['optimizer']['type'], 'args_d' : {'lr':0.0},'args_g' : {'lr':0.0}}, s_dict['regulate_path_length'])
        self.__optimG.load_state_dict(s_dict['optimizer']['state_dict_optimG'])
        self.__optimD.load_state_dict(s_dict['optimizer']['state_dict_optimD'])
        self.__total_epochs = s_dict['epochs']
        self.__total_G_iter = s_dict['iter_of_G']
        self.__total_D_iter = s_dict['iter_of_D']
        self.__total_G_losses = s_dict['loss_of_G']
        self.__total_D_losses = s_dict['loss_of_D']
        self.lazy_reg = s_dict['lazy_regularization']
        self.n_critic = s_dict['n_critic']

    def load(self, path):
        s_dict = torch.load(path)
        self.load_from_state_dict(s_dict)
        
    @property
    def device(self):
        return self.__device
    
    @device.setter
    def device(self,device):
        if self.__G is not None and self.__D is not None:
            self.__device = utils.set_device(device)
            self.__G.to(self.__device)
            self.__D.to(self.__device)
            plr.device = self.__device


    @property
    def total_iter(self):
        return {'D' : self.__total_D_iter, 'G' : self.__total_G_iter}

    @property
    def total_epochs(self):
        return self.__total_epochs

    @property
    def G(self):
        return self.__G

    @property
    def D(self):
        return self.__D
        
    @property
    def loss(self):
        return {'loss criterion of G' : self.__lossG, 'loss criterion of D' : self.__lossD}

    @property
    def optimizer(self):
        return {'optimizer type' : self.__optim_type, 'optim_G' : self.__optimG.state_dict(),'optim_D' : self.__optimD.state_dict()}

    @optimizer.setter
    def optimizer(self,optimizer):
        self.__optim_type = optimizer['type']
        if optimizer['type']=='lsgan':
            self.__optimG = optim.Adam(self.__G.parameters(), **optimizer['args_g'])
            self.__optimD = optim.Adam(self.__D.parameters(), **optimizer['args_d'])
            self.__lossG, self.__lossD = nn.MSELoss(), nn.MSELoss()
        if optimizer['type']=='logistic_loss':
            self.__optimG = optim.Adam(self.__G.parameters(),**optimizer['args_g'])
            self.__optimD = optim.Adam(self.__D.parameters(), **optimizer['args_d'])
            self.__lossG, self.__lossD = 'logistic_G', 'logistic_D'
        if optimizer['type']=='logistic_loss_r1':
            self.__optimG = optim.Adam(self.__G.parameters(),**optimizer['args_g'])
            self.__optimD = optim.Adam(self.__D.parameters(), **optimizer['args_d'])
            self.__lossG, self.__lossD = 'logistic_G', 'logistic_D_r1'
        if optimizer['type']=='logistic_loss_ns':
            self.__optimG = optim.Adam(self.__G.parameters(),**optimizer['args_g'])
            self.__optimD = optim.Adam(self.__D.parameters(), **optimizer['args_d'])
            self.__lossG, self.__lossD = 'logistic_G', 'logistic_D_ns'
    
    def make_latent_z(self, batch_size, n_latents, clip_value=None, style_mix_step=[]):
        out1 = [torch.randn(batch_size, n_latents, device=self.__device)] if style_mix_step==[] else [torch.randn(batch_size, n_latents, device=self.__device) for _ in range(2)]
        out2 = torch.randn(batch_size, 1, device=self.__device)
        if isinstance(clip_value, tuple):
            out1 = [torch.clamp(_out1, *clip_value) for _out1 in out1]
            out2 = torch.clamp(out2, *clip_value)
        return out1, out2, self.make_b_noise(batch_size), style_mix_step

    def make_b_noise(self, batch_size):
        b_std = 0.1
        return torch.randn(batch_size)*b_std

    
    def eval(self, n_fakes, clip_value=None, style_mix_step=[], save_path=None, **show_image_kwargs):
        with torch.no_grad():
            fake = self.out_G(n_fakes, clip_value, style_mix_step).detach().cpu()
            dset.show_images_from_tensors(fake, **show_image_kwargs)
            if save_path != None and os.path.exists(save_path):
                dset.save_images_from_tensors(fake, save_path, **show_image_kwargs)

    def G_zerograd(self):
        self.__G.zero_grad()

    def D_zerograd(self):
        self.__D.zero_grad()

    def out_G(self, n_fakes, clip_value=None, style_mix_step=[]):
        return self.__G(*self.make_latent_z(n_fakes, self.__G.dim_latent, clip_value, style_mix_step))

    def out_D(self, real_sample):
        real_sample = real_sample.to(self.__device)
        return self.__D(real_sample)

    def optimG_step(self):
        self.__optimG.step()

    def optimD_step(self):
        self.__optimD.step()

    def confirm_iter_D(self):
        self.__total_D_iter = self.__total_D_iter + 1

    def confirm_iter_G(self):
        self.__total_G_iter = self.__total_G_iter + 1
    
    def __call__(self, epochs, eval_loss=1,eval_D=1, eval_G=1,eval_dict={'n_fakes':16,'save_path':None,'grid_size':(4,4)}):

        dataloader = dset.create_image_loader_from_path(self.image_path, self.image_size, self.batch_size)
        num_batchs = len(dataloader)
        d_loss_epochs = []
        g_loss_epochs = []
        #D_x, D_g1, D_g2 = [], [], []
        for epoch in tqdm(range(epochs), desc='epochs', postfix=f'total epochs : {self.total_epochs}'):
            for i,data in tqdm(enumerate(dataloader,0),leave=False,desc=f"batches, total iter:{self.total_iter}"):
                
                torch.autograd.set_detect_anomaly(True)
                #self.__optimD.zero_grad()
                self.__D.zero_grad()
                real_sample = data[0].to(self.__device)
                loss_D = loss.D(self, real_sample)
                loss_D.backward()
                d_loss_epochs = d_loss_epochs + [loss_D.item()]
                self.__total_D_losses = self.__total_D_losses + [loss_D.item()]
                self.__optimD.step()
                self.confirm_iter_D()

                if self.__total_D_iter % self.n_critic == 0:
                    reg = None
                    #self.__optimG.zero_grad()
                    self.__G.zero_grad()
                    loss_G = loss.G(self)
                    g_loss_epochs = g_loss_epochs + [loss_G.item()]
                    self.__total_G_losses = self.__total_G_losses + [loss_G.item()]
                    if self.__reg_path_len:
                        if (self.lazy_reg > 0 and self.__total_G_iter % self.lazy_reg == 0):
                            reg = plr(self.__G, self.batch_size)
                            print(" pl : ", reg.item())
                    if reg is not None:
                        loss_G = loss_G.clone() + reg
                    loss_G.backward()
                    self.confirm_iter_G()
                    self.__optimG.step()

                if self.__total_D_iter % eval_loss == 0 or i == (num_batchs - 1):
                    mean_loss_D = np.mean(d_loss_epochs)
                    mean_loss_G = np.mean(g_loss_epochs)
                    print('\nmean loss of D : ', mean_loss_D)
                    print('mean loss of G : ', mean_loss_G)
                    d_loss_epochs=[]
                    g_loss_epochs=[]

                if self.__total_D_iter % eval_D == 0 or i == (num_batchs - 1):
                    mean_D_real = np.mean(self.out_D(real_sample).detach().cpu().numpy())
                    mean_D_fake = np.mean(self.out_D(self.out_G(real_sample.size(0)).detach()).detach().cpu().numpy())
                    print('\nmean D(x) : ', mean_D_real)
                    print('mean D(g) : ', mean_D_fake)

                '''
                if self.__device.type=='cuda' and self.__total_D_iter % self.empty_cache == 0:
                    torch.cuda.empty_cache()
                '''

            if eval_G > 0 and epoch % eval_G == 0:
                self.eval(**eval_dict)


sys.modules[__name__] = __train()


if __name__=='__main__':
    tr = __train()
    tr.image_size = 32
    
    img_channels=[32,16,8,4]
    print(img_channels)
    G = SG2_Generator(32,img_channels,256,4)
    img_channels.reverse()
    D = SG2_Discriminator(32,img_channels)
    print(img_channels)
    tr.init(G,D,{'type':'logistic_loss','args_d':{'lr':1e-3},'args_g':{'lr':1e-3}})
    tr.image_path = '/home/shy/kiana_resized/'
    tr.image_path = 'p/'
    tr.batch_size = 4
    tr.lazy_reg = 8
    epochs = 200
    evalG = 20
    for epoch in range(epochs):
        tr(1,eval_G=0,eval_dict={'n_fakes':16,'save_path':None,'grid_size':(4,4),'max_size':128})
        if epoch % evalG == 0 or epoch == epochs-1:
            tr.eval(16,clip_value=(-1.0,1.0),grid_size=(4,4))