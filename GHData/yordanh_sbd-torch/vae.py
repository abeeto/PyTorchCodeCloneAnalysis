import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, alpha=1, beta=1, latent_n=1, device="cpu", sbd=True):
        super(VAE, self).__init__()
        self.latent_n = latent_n
        self.device = device
        self.sbd = sbd
        
        self.alpha = alpha
        self.beta = beta
        
        self.conv_ch = [32,32,64,64]
        self.dense_ch = [1024, 32]
        self.sbd_conv_ch = [64, 64]

        # IMAGE ENCODER
        k_size=3
        self.encoder_conv_0 = nn.Conv2d(3, self.conv_ch[0], k_size, padding=1, stride=2) # (32, 32)
        self.encoder_conv_1 = nn.Conv2d(self.conv_ch[0], self.conv_ch[1], k_size, padding=1, stride=2) # (16, 16)        
        self.encoder_conv_2 = nn.Conv2d(self.conv_ch[1], self.conv_ch[2], k_size, padding=1, stride=2) # (8, 8)
        self.encoder_conv_3 = nn.Conv2d(self.conv_ch[2], self.conv_ch[3], k_size, padding=1, stride=2) # (4, 4)
        
        self.encoder_dense_0 = nn.Linear(self.dense_ch[0], self.dense_ch[1])
        self.encoder_mu = nn.Linear(self.dense_ch[1], self.latent_n)
        self.encoder_ln_var = nn.Linear(self.dense_ch[1], self.latent_n)
        
        # IMAGE DECONV DECODER
        k_size = 4
        self.deconv_dense_0 = nn.Linear(self.latent_n, self.dense_ch[1])
        self.deconv_dense_1 = nn.Linear(self.dense_ch[1], self.dense_ch[0])
        self.deconv_conv_3 = nn.ConvTranspose2d(self.conv_ch[3], self.conv_ch[2], k_size, stride=2, padding=1)
        self.deconv_conv_2 = nn.ConvTranspose2d(self.conv_ch[2], self.conv_ch[1], k_size, stride=2, padding=1)
        self.deconv_conv_1 = nn.ConvTranspose2d(self.conv_ch[1], self.conv_ch[0], k_size, stride=2, padding=1)
        self.deconv_output_img = nn.ConvTranspose2d(self.conv_ch[0], 3, k_size, stride=2, padding=1)
        
        # IMAGE SPATIAL DECODER
        k_size = 3
        self.sbd_conv_2 = nn.Conv2d(self.latent_n+2, self.sbd_conv_ch[0], k_size, padding=1)
        self.sbd_conv_1 = nn.Conv2d(self.sbd_conv_ch[0], self.sbd_conv_ch[1], k_size, padding=1)
        self.sbd_conv_0 = nn.Conv2d(self.sbd_conv_ch[1], 3, k_size, padding=1)
        
        self.image_size = 64
        a = np.linspace(-1, 1, self.image_size)
        b = np.linspace(-1, 1, self.image_size)
        x, y = np.meshgrid(a, b)
        x = x.reshape(self.image_size, self.image_size, 1)
        y = y.reshape(self.image_size, self.image_size, 1)
        self.xy = np.concatenate((x,y), axis=-1)

        self.encoder = [self.encoder_conv_0,
                        self.encoder_conv_1,
                        self.encoder_conv_2,
                        self.encoder_conv_3,
                        self.encoder_dense_0,
                        self.encoder_mu,
                        self.encoder_ln_var]
        
        self.deconv_decoder = [self.deconv_dense_0,
                               self.deconv_dense_1,
                               self.deconv_conv_3,
                               self.deconv_conv_2,
                               self.deconv_conv_1,
                               self.deconv_output_img]

        self.sbd_decoder = [self.sbd_conv_2,
                            self.sbd_conv_1,
                            self.sbd_conv_0]
        
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.encoder)):
            self.encoder[i].weight.data.normal_(0, 0.01)
            
        for i in range(len(self.deconv_decoder)):
            self.deconv_decoder[i].weight.data.normal_(0, 0.01)

        for i in range(len(self.sbd_decoder)):
            self.sbd_decoder[i].weight.data.normal_(0, 0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        conv_0_encoded = F.leaky_relu(self.encoder_conv_0(x))
        conv_1_encoded = F.leaky_relu(self.encoder_conv_1(conv_0_encoded))
        conv_2_encoded = F.leaky_relu(self.encoder_conv_2(conv_1_encoded))
        conv_3_encoded = F.leaky_relu(self.encoder_conv_3(conv_2_encoded))

        reshaped_encoded = torch.flatten(conv_3_encoded, start_dim=1)
        dense_0_encoded = F.leaky_relu(self.encoder_dense_0(reshaped_encoded))
        mu = self.encoder_mu(dense_0_encoded)
        logvar = self.encoder_ln_var(dense_0_encoded)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def deconv_decode(self, z):
        dense_0_decoded = F.leaky_relu(self.deconv_dense_0(z))
        dense_1_decoded = F.leaky_relu(self.deconv_dense_1(dense_0_decoded))
        reshaped_decoded = dense_1_decoded.view((len(dense_1_decoded), self.conv_ch[-1], 4, 4))
        deconv_3_decoded = F.leaky_relu(self.deconv_conv_3(reshaped_decoded))
        deconv_2_decoded = F.leaky_relu(self.deconv_conv_2(deconv_3_decoded))
        deconv_1_decoded = F.leaky_relu(self.deconv_conv_1(deconv_2_decoded))
        out_img = self.deconv_output_img(deconv_1_decoded)
        
        return torch.sigmoid(out_img)

    def sbd_decode(self, z):
        batchsize = len(z)
        xy_tiled = torch.from_numpy(np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(self.device)
        
        z_tiled = torch.repeat_interleave(z, self.image_size*self.image_size, dim=0).view(batchsize, self.image_size, self.image_size, self.latent_n)
        
        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        sbd_2_decoded = F.leaky_relu(self.sbd_conv_2(z_and_xy))
        sbd_1_decoded = F.leaky_relu(self.sbd_conv_1(sbd_2_decoded))
        out_img = self.sbd_conv_0(sbd_1_decoded)
        
        return torch.sigmoid(out_img)
    
    def get_latent(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        return z
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)    
        if self.sbd:
            img_out = self.sbd_decode(z)
        else:
            img_out = self.deconv_decode(z)
        
        return img_out, mu, logvar 
    
    def get_loss(self):
        def loss(img_in, img_out, mu, logvar):
            rec = nn.MSELoss(reduction="none")(img_out, img_in)
            rec = torch.mean(torch.sum(rec.view(rec.shape[0], -1), dim=-1))
            
            kld = (((-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / img_in.shape[0])

            rec *= self.alpha
            kld *= self.beta

            return rec + kld, rec, kld
    
        return loss

