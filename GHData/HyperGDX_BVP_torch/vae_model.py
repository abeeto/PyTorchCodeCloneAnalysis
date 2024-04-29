import torch
import torch.nn as nn
import numpy as np


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return self.relu(out)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, output_padding=1):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convt1(x)
        out = self.bn1(out)
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self, in_c=3, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        self.in_c = in_c

        self.conv1 = DownBlock(in_c, 10, kernel_size=9, stride=1)
        self.conv2 = DownBlock(10, 20, kernel_size=7, stride=3)
        self.conv3 = DownBlock(20, 40, kernel_size=5, stride=1)
        self.conv4 = DownBlock(40, 80, kernel_size=3, stride=3)  # (b, 80, 4, 4)
        self.dropout = nn.Dropout2d(0.5)
        self.fc5 = nn.Linear(1280, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(1024, self.z_dim)

    def forward(self, x):
        """forward pass for encoder.
        x: img with (3, 64, 64)
        """
        b = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = out.view(b, -1)
        out = self.fc5(out)
        out = self.relu(out)
        return self.fc6(out)


class Decoder(nn.Module):
    def __init__(self, out_c=3, z_dim=512):
        super().__init__()
        self.out_c = out_c
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1280)

        self.convt3 = UpBlock(80, 40, kernel_size=3, stride=3, output_padding=1)
        self.convt4 = UpBlock(40, 20, kernel_size=5, stride=1, output_padding=0)
        self.convt5 = UpBlock(20, 10, kernel_size=7, stride=3, output_padding=1)
        #self.convt6 = UpBlock(10, out_c, kernel_size=9, stride=1, output_padding=0)
        self.convt6 = nn.ConvTranspose2d(10, out_c, kernel_size=9, stride=1, bias=False)
        self.tanh = nn.Tanh()  # NOTE: Different from the VideoVAE paper

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)

        out = out.view(-1, 80, 4, 4)  # the shape from encoder before its first FC layers
        out = self.convt3(out)
        out = self.convt4(out)
        out = self.convt5(out)
        out = self.convt6(out)
        return self.tanh(out)


class AttributeNet(nn.Module):
    def __init__(self, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_act = n_act
        self.n_id = n_id

        self.fc1 = nn.Linear(z_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc_act = nn.Linear(self.h_dim, n_act)
        self.fc_id = nn.Linear(self.h_dim, n_id)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        logit_act = self.fc_act(out)
        logit_id = self.fc_id(out)
        return logit_act, logit_id


class AttributeNet_v2(nn.Module):
    def __init__(self, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_act = n_act
        self.n_id = n_id

        self.fc1_act = nn.Linear(z_dim, self.h_dim)
        self.fc1_id = nn.Linear(z_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2_act = nn.Linear(self.h_dim, n_act)
        self.fc2_id = nn.Linear(self.h_dim, n_id)

    def forward(self, x):
        h1_act = self.relu(self.fc1_act(x))
        h1_id = self.relu(self.fc1_id(x))
        logit_act = self.fc2_act(h1_act)
        logit_id = self.fc2_id(h1_id)

        return logit_act, logit_id

# ref: https://github.com/pytorch/examples/blob/master/vae/main.py


class DistributionNet(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, out_dim=512):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc21 = nn.Linear(h_dim, out_dim)
        self.fc22 = nn.Linear(h_dim, out_dim)

    # follow the term in VAE. check: https://github.com/pytorch/examples/blob/master/vae/main.py#L49-L51
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class MLP(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, out_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)

########################################
#                 Ours                 #
########################################


class ConditionalBatchNorm(nn.Module):
    def __init__(self, shape, n_features=3):
        super().__init__()
        self.n_features = n_features
        self.shape = shape
        self.eps = 1e-5

        n1 = np.prod(np.array(list(shape))) * n_features
        self.fc1 = nn.Linear(n1, 9)
        self.relu = nn.ReLU(inplace=True)
        self.fc2_beta = nn.Linear(9, self.n_features)
        self.fc2_gamma = nn.Linear(9, self.n_features)

    def forward(self, x):
        assert x.size(1) == self.n_features

        b = x.size(0)
        # get mu, std per channel
        #mu  = x.permute(1, 0, 2, 3).view(b, self.n_features, -1).mean(dim=2, keepdim=True)
        mu = x.permute(1, 0, 2, 3).contiguous().view(self.n_features, -1).mean(dim=1)
        std = x.permute(1, 0, 2, 3).contiguous().view(self.n_features, -1).std(dim=1)
        # TODO BatchNorm
        #mu  = x.view(b, self.n_features, -1).mean(dim=2, keepdim=True)
        #std = x.view(b, self.n_features, -1).std(dim=2, keepdim=True)
        h1 = self.relu(self.fc1(x.view(b, -1)))
        beta = self.fc2_beta(h1)    # (b, n_features)
        gamma = self.fc2_gamma(h1)  # (b, n_features)

        out = (x - mu[None, :, None, None]) / (std[None, :, None, None] + self.eps) * gamma[:, :, None, None] + beta[:, :, None, None]

        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, bias=False, norm='CBN', shape=None):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if norm == 'CBN':
            self.bn = ConditionalBatchNorm(shape, n_features=out_c)
        else:
            self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# encode style, i.e. identity


class StyleEncoder(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()

        self.conv1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.conv2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.conv3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

# encode content, i.e. action


class ContentEncoder(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()

        self.conv1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.conv2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.conv3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class OurDecoder(nn.Module):
    def __init__(self, in_c=64):
        super().__init__()

        self.convt1 = ConvBNReLU(3, 8, kernel_size=3, shape=(28, 28))
        self.convt2 = ConvBNReLU(8, 16, kernel_size=3, shape=(28, 28))
        self.convt3 = ConvBNReLU(16, 32, kernel_size=3, shape=(28, 28))

    def forward(self, x):
        out = self.convt1(x)
        out = self.convt2(out)
        out = self.convt3(out)
        return out


class MotionNet(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 512
        hidden_size = 128
        num_layers = 1
        bidirectional = False

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

    def forward(self, x):

        return x


class Classifier(nn.Module):
    def __init__(self, in_c=3, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.in_c = in_c
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.n_act = n_act
        self.n_id = n_id

        self.enc = Encoder(in_c=in_c, z_dim=z_dim)
        self.attr_net = AttributeNet(z_dim=z_dim, h_dim=128, n_act=n_act, n_id=n_id)

    def forward(self, x):
        x_enc = self.enc(x)
        out_act, out_id = self.attr_net(x_enc)
        return out_act, out_id


class VideoVAE(nn.Module):
    def __init__(self, z_dim=512, h_dim=512, n_act=10, n_id=9, input_size=512*3, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.n_act = n_act
        self.n_id = n_id

        # Encoder and attr_net can be trained indepently
        self.enc = Encoder()
        self.attr_net = AttributeNet(z_dim=z_dim, h_dim=128, n_act=n_act, n_id=n_id)

        # Conditional Approximate Posterior
        self.post_q = DistributionNet(in_dim=z_dim,
                                      h_dim=128,
                                      out_dim=512)
        self.post_a = DistributionNet(in_dim=(z_dim*2+n_act+n_id),
                                      h_dim=128,
                                      out_dim=512)
        self.post_dy = DistributionNet(in_dim=(z_dim+z_dim+z_dim),
                                       h_dim=128,
                                       out_dim=512)

        self.mlp_lstm = MLP(in_dim=512, h_dim=128, out_dim=512)

        # Prior
        self.prior = DistributionNet(in_dim=(z_dim+n_act+n_id),
                                     h_dim=128,
                                     out_dim=512)

        # Decoder
        self.dec = Decoder()

        # LSTM
        self.input_size = input_size  # dimension: (z_dim+z_dim*2),  z_t and [mu_q, logvar_q]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # inputs:  input, (h_0, c_0)
        # outputs: output, (h_n, c_n)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False)

        if self.bidirectional:
            self.lstm_backward = nn.LSTM(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         bidirectional=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def pred_attr(self, x):
        """Classify the action and identity."""
        x_enc = self.enc(x)
        logit_act, logit_id = self.attr_net(x_enc)
        _, pred_act = logit_act.max(1)
        _, pred_id = logit_id.max(1)
        return pred_act, pred_id

    def seq_cls(self, img_seq):
        """ Classify the whole sequence to get the attributes. """
        b, t, c, h, w = img_seq.shape
        img_seq_expand = img_seq.view(-1, c, h, w)
        pred_act, pred_id = self.pred_attr(img_seq_expand)

        # view back to b, t
        pred_act, pred_id = pred_act.view(b, t), pred_id.view(b, t)

        # voting
        pred_act_mode, _ = torch.mode(pred_act, dim=1, keepdim=True)
        pred_id_mode, _ = torch.mode(pred_id, dim=1, keepdim=True)

        # repeat
        pred_act_mode = pred_act_mode.repeat(1, 10)
        pred_id_mode = pred_id_mode.repeat(1, 10)

        # vutils.save_image(img_seq_expand*.5 + .5, 'hey.jpg', nrow=10)
        return pred_act, pred_id

    def cond_approx_posterior(self, x_enc, a_label, phi_h):
        # For Posterior q
        mu_q, logvar_q = self.post_q.encode(x_enc)

        # For Posterior a
        #   phi_q: [mu_q, logvar_q]
        #   merge: (phi_q, a_label)
        #   phi_q_merged: (phi_q, a_label)
        phi_q_merged = torch.cat([mu_q, logvar_q, a_label], dim=1)  # NOTE: merge along z-axis
        mu_a, logvar_a = self.post_a.encode(phi_q_merged)

        # For Posterior dy
        #   phi_a: [mu_a, logvar_a, ]
        #   merge: (phi_a, \phi(h_{t-1}))
        phi_a_merged = torch.cat([mu_a, logvar_a, phi_h[0]], dim=1)
        z_dy, mu_dy, logvar_dy = self.post_dy(phi_a_merged)

        return mu_q, logvar_q, z_dy, mu_dy, logvar_dy

    def lstm_forward(self, mu_q, logvar_q, z_t, h_prev, c_prev):
        """LSTM forward for 1 time step.
        This only propagate 1 time step, so lstm_output should be the same as h_t

        params:
            - mu_q, logvar_q
            - z_t: samples from prior (z_p) or posterior (z_dy)
            - h_prev: hidden states from t-1
            - c_prev: cell states from t-1
        returns:
            - lstm_output: the output of LSTM. It contains all the *h_t* for eatch t. Here we have t=1, thus it should be equal to h_t
            - h_t: contains the *last* hiddent states for each time step. (n_layers*)
            - c_t: cell states for t
        notes:
            - lstm_input: merged of (z_t, mu_q, logvar_q).
                          Has to be shape: (seq_len=1, batch_size, inputdim=z_dim+z_dim*2)
        """

        lstm_input = torch.cat([z_t, mu_q, logvar_q], dim=1)
        lstm_input = lstm_input.unsqueeze(dim=0)
        lstm_output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))
        # assert (z_t - h_t).sum() == 0
        return lstm_output, h_t, c_t

    def forward(self, x_t, pred_act, pred_id, h_prev, c_prev):
        """Forward pass for the VideoVAE.

        The model first encode x_t into structured latent space, then sample z_t from either the 
        prior distribution (\phi_p at test time) or posterior distribution (\phi_dy at training time)
        to reconstruct x_t, with a LSTM to model the temporal information.

        For the details, please see Fig. 2 at page 5:
        https://arxiv.org/pdf/1803.08085.pdf
        """

        batch_size = x_t.size(0)

        # NOTE: no_grad here
        with torch.no_grad():
            x_enc = self.enc(x_t)

        # to one-hot
        a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(x_enc)
        a_label[torch.arange(batch_size), pred_act] = 1
        a_label[torch.arange(batch_size), pred_id+self.n_act] = 1  # id: 0 -> 0 + n_act. for one-hot representation

        # transformed the h_prev
        phi_h = self.mlp_lstm(h_prev)

        # For Conditional Approximate Posterior. Check page 7 in https://arxiv.org/pdf/1803.08085.pdf
        mu_q, logvar_q, z_dy, mu_dy, logvar_dy = self.cond_approx_posterior(x_enc, a_label, phi_h)

        # For Prior p
        #   phi_h_merge: [phi_h, a]
        phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
        z_p, mu_p, logvar_p = self.prior(phi_h_merged)

        # In training, we sample from phi_dy to get z_t
        z_t = z_dy

        # LSTM forward
        lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

        # Reconstruction
        recon_x_t = self.dec(z_t)

        return recon_x_t, z_t, lstm_output, [h_t, c_t], [mu_p, logvar_p], [mu_dy, logvar_dy]

    def synthesize(self, x_prev, h_prev, c_prev, holistic_attr, only_prior=True, is_first_frame=True):
        """Synthesize the sequences. 

        Setting 1: Holistic attribute controls only. We only generate frames from prior distribution. (tend to get more blurry results.)
        Setting 2: Holistic attr. controls & first frame. The first frame is provided. Hence the generated first frame 
                   is the reconstruction of the given frame.

        For the details, please see sec. 4.2 at page 9:
        https://arxiv.org/pdf/1803.08085.pdf
        """

        holistic_act = holistic_attr['action']
        holistic_id = holistic_attr['identity']

        batch_size = 1  # test batch_size
        ##################################
        #    synthesize for setting 1    #
        ##################################
        if only_prior:
            # transformed the h_prev
            phi_h = self.mlp_lstm(h_prev)
            a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(h_prev)

            for i in range(batch_size):
                a_label[i, holistic_act] = 1
                a_label[i, self.n_act+holistic_id] = 1

            if is_first_frame:
                # For Prior
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                x_gen = self.dec(z_p) * 0.5 + 0.5

                return x_gen, h_prev, c_prev
            else:
                # x_t: x_gen from previous step
                x_enc = self.enc(x_prev)

                # q
                mu_q, logvar_q = self.post_q.encode(x_enc)

                # p
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                z_t = z_p

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

                return x_gen, h_t, c_t
        ##################################
        #    synthesize for setting 2    #
        ##################################
        else:
            # x_t: x_gen from previous step
            x_enc = self.enc(x_prev)

            # q
            mu_q, logvar_q = self.post_q.encode(x_enc)

            # transformed the h_prev
            phi_h = self.mlp_lstm(h_prev)

            # attr: should be control by the first frame provided by us
            # First frame in this setting is the reconstruction of the input
            a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(x_enc)

            for i in range(batch_size):
                # should specify the act here
                # should not change id here
                a_label[i, holistic_act] = 1
                a_label[i, self.n_act + holistic_id] = 1

            if is_first_frame:

                mu_q, logvar_q, z_dy, mu_dy, logvar_dy = self.cond_approx_posterior(x_enc, a_label, phi_h)

                z_t = z_dy

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

            # attr: should be control by the holistic_attr provided by us
            else:
                # p
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                z_t = z_p

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

            return x_gen, h_t, c_t

    def reset(self, batch_size=64, reset='zeros'):
        """ reset lstm state.

        Returns:
            h_0, c_0 for LSTM.
        """
        use_cuda = next(self.parameters()).is_cuda

        h_0 = torch.zeros(1, batch_size, self.z_dim)
        c_0 = torch.zeros(1, batch_size, self.z_dim)

        # should set to random if we are synthesizing using only prior distribution.
        if reset == 'random':
            h_0 = torch.randn_like(h_0)
            c_0 = torch.randn_like(c_0)

        if use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return h_0, c_0

    def load_cls_net(self, weight_path):
        state_dict = torch.load(weight_path)
        self.enc.load_state_dict(state_dict['encoder'])
        self.attr_net.load_state_dict(state_dict['attr_net'])
