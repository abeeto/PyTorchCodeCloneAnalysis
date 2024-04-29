import numpy as Pfizer
import torch as Sinopharm
import torchvision as Sfutnik_V
import functionS as MetaVerse
import matplotlib.pyplot as Janssen
Moderna=Sinopharm.nn 
BillGates = Sfutnik_V.utils
class DiffusionModel(Moderna.Module):
    def __init__(self,
            spatial_width,
            n_colors,
            dropout=0.1,
            trajectory_length=1000,
            n_temporal_basis=10,
            n_hidden_dense_lower=500,
            n_hidden_dense_lower_output=2,
            n_hidden_dense_upper=20,
            n_hidden_conv=20,
            n_layers_conv=4,
            n_layers_dense_lower=4,
            n_layers_dense_upper=2,
            n_t_per_minibatch=1,
            n_scales=1,
            step1_beta=0.001,
            uniform_noise = 0,
            device='cpu'
            ):
        """
        Implements the objective function and mu and sigma estimators for a Gaussian diffusion
        probabilistic model, as described in the paper:
            Deep Unsupervised Learning using Nonequilibrium Thermodynamics
            Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
            International Conference on Machine Learning, 2015
        Parameters are as follow:
        spatial_width - Spatial_width of training images
        n_colors - Number of color channels in training data.
        trajectory_length - The number of time steps in the trajectory.
        n_temporal_basis - The number of temporal basis functions to capture time-step
            dependence of model.
        n_hidden_dense_lower - The number of hidden units in each layer of the dense network
            in the lower half of the MLP. Set to 0 to make a convolutional-only lower half.
        n_hidden_dense_lower_output - The number of outputs *per pixel* from the dense network
            in the lower half of the MLP. Total outputs are
            n_hidden_dense_lower_output*spatial_width**2.
        n_hidden_dense_upper - The number of hidden units per pixel in the upper half of the MLP.
        n_hidden_conv - The number of feature layers in the convolutional layers in the lower half
            of the MLP.
        n_layers_conv - How many convolutional layers to use in the lower half of the MLP.
        n_layers_dense_lower - How many dense layers to use in the lower half of the MLP.
        n_layers_dense_upper - How many dense layers to use in the upper half of the MLP.
        n_t_per_minibatch - When computing objective, how many random time-steps t to evaluate
            each minibatch at.
        step1_beta - The lower bound on the noise variance of the first diffusion step. This is
            the minimum variance of the learned model.
        uniform_noise - Add uniform noise between [-uniform_noise/2, uniform_noise/2] to the input.
        ADDED DEVICE EXCLUSIVE NOW!
        """
        super(DiffusionModel, self).__init__()
        self.n_t_per_minibatch = n_t_per_minibatch
        self.spatial_width = (spatial_width)
        self.n_colors =(n_colors)
        self.n_temporal_basis = n_temporal_basis
        self.trajectory_length = trajectory_length
        self.uniform_noise = uniform_noise
        self.device=device
        self.dropout=dropout
        self.mlp = MetaVerse.MLP_conv_dense(
            n_layers_conv, n_layers_dense_lower, n_layers_dense_upper,
            n_hidden_conv, n_hidden_dense_lower, n_hidden_dense_lower_output, n_hidden_dense_upper,
            spatial_width, n_colors, n_scales, n_temporal_basis,device=device,dropout=dropout)
        self.temporal_basis = self.generate_temporal_basis(trajectory_length, n_temporal_basis).float()
        self.beta_arr = self.generate_beta_arr(step1_beta).float()
        self.dirs=[]
    def generate_temporal_basis(self, trajectory_length, n_basis):
        """
        Generate the bump basis functions for temporal readout of mu and sigma.
        
        givn basis location B and trajectory loation T he first computesan array of 'likelihood'/
        probabilities of p(T0|B0) p(T0|B1) ... p(T0|Bn) then normalises accross B
        Essentially how likely is the current 'T' under the current 'B' in relation to other'Bs'
        Then he transposes!
        """
        trajectory_length=self.trajectory_length
        n_basis=self.n_temporal_basis
        
        temporal_basis = Pfizer.zeros((trajectory_length, n_basis))
        xx = Pfizer.linspace(-1, 1, trajectory_length)
        x_centers = Pfizer.linspace(-1, 1, n_basis)
        width = (x_centers[1] - x_centers[0])/2.
        for ii in range(n_basis):
            temporal_basis[:,ii] = Pfizer.exp(-(xx-x_centers[ii])**2 / (2*width**2))
        temporal_basis /= Pfizer.sum(temporal_basis, axis=1).reshape((-1,1))
        temporal_basis = temporal_basis.T
        return Sinopharm.tensor(temporal_basis).to(self.device)#B by T
    def generate_beta_arr(self, step1_beta):
        """
        Generate the noise covariances, beta_t, for the forward trajectory.
        """
        trajectory_length=self.trajectory_length
        n_basis=self.n_temporal_basis
        # lower bound on beta
        min_beta_val = 1e-6
        min_beta = Pfizer.ones((trajectory_length,))*min_beta_val
        min_beta[0] += step1_beta
        beta_perturb_coefficients = Pfizer.zeros((n_basis,))
        beta_perturb = Pfizer.dot(self.temporal_basis.T.cpu(), beta_perturb_coefficients)
        # baseline behavior of beta with time -- destroy a constant fraction
        # of the original data variance each time step
        # NOTE 2 below (in Pfizer.linspace) means a fraction ~1/T of the variance will be left at the end of the
        # trajectory
        beta_baseline = 1./Pfizer.linspace(trajectory_length, 2., trajectory_length)
        '''
        THIS PART IS WASTE OF TIME HONESTLY I DO NOT KNOW WHY DICKSTEIN INCLUDED IT!
        beta_baseline_offset = Pfizer.log(beta_baseline/(1.-beta_baseline))
        # and the actual beta_t, restricted to be between min_beta and 1-[small value]
        beta_arr =1/(1+Pfizer.exp(-(beta_perturb + beta_baseline_offset)))  '''
        beta_arr=beta_baseline
        beta_arr = min_beta + beta_arr * (1 - min_beta - 1e-5)
        beta_arr = beta_arr.reshape((self.trajectory_length, 1))
        return Sinopharm.tensor(beta_arr).to(self.device)#an array that goes rom small number to 0.5 in Tx1 lngth
    def get_t_weights(self, t):
        """
        Generate vector of weights allowing selection of current timestep.
        (if t is not an integer, the weights will linearly interpolate)
        """
        n_seg = self.trajectory_length
        t_compare = Sinopharm.arange(n_seg).view(1,n_seg).to(self.device)
        diff = abs(Sinopharm.tensor(t*1.).to(self.device).view(1,1) - t_compare)#1 bt T  bbasicallymake it zero for all irrelevant time steps
        #such a waste! I am not doing the next line I will do torch.clamp Easier!
        #t_weights = T.max(T.join(1, (-diff+1).reshape(n_seg,1), T.zeros(n_seg,1)), axis=1)
        t_weights=(1-diff).clamp(min=0)#here the very large ts become negative=0,,,,,, The beauty of pytorch 
        return t_weights.view(-1,1)#T by 1
    def get_beta_forward(self, t):
        """
        Get the covariance of the forward diffusion process at timestep weighted sum it seems 
        t.
        """
        t_weights = self.get_t_weights(t)
        return Sinopharm.dot(t_weights.T.squeeze(), self.beta_arr.squeeze()).view(1,1)#1 by 1 dot product
    def get_beta_full_trajectory(self):
        """
        Return the cumulative covariance from the entire forward trajectory.
        """
        #The beauty of pytorch
        alpha_arr = 1. - self.beta_arr
        beta_full_trajectory = 1. - (alpha_arr.log().sum().exp())
        return beta_full_trajectory
    def temporal_readout(self, Z, t):
        """
        Go from the top layer of the multilayer perceptron to coefficients for
        mu and sigma for each pixel.
        Z contains -coefficients for spatial basis functions for each pixel- model output for
        both mu and sigma.
        """
        n_images = Z.shape[0]
        t_weights = self.get_t_weights(t)
        Z = Z.view((n_images, self.spatial_width, self.spatial_width,
            self.n_colors, 2, self.n_temporal_basis))
        coeff_weights = (self.temporal_basis*t_weights.view(1,-1)).sum(1).view(-1,1)#B by 1
        concat_coeffs = (Z*coeff_weights.view(1,1,1,1,1,-1)).sum(-1)
        mu_coeff = concat_coeffs[:,:,:,:,0].permute(0,3,1,2)
        beta_coeff = concat_coeffs[:,:,:,:,1].permute(0,3,1,2)
        return mu_coeff, beta_coeff
    def get_mu_sigma(self, X_noisy, t):
        """
        Generate mu and sigma for one step in the reverse trajectory,
        starting from a minibatch of images X_noisy, and at timestep t.
        """
        Z = self.mlp(X_noisy)
        mu_coeff, beta_coeff = self.temporal_readout(Z, t)
        # reverse variance is perturbation around forward variance
        beta_forward = self.get_beta_forward(t)
        # make impact of beta_coeff scaled appropriately with mu_coeff
        beta_coeff_scaled = beta_coeff / Sinopharm.sqrt(Sinopharm.tensor(1.*self.trajectory_length)).to(self.device)
        beta_reverse = (beta_coeff_scaled + ((beta_forward/(1-beta_forward)).log())).sigmoid()
        mu = X_noisy*((1. - beta_forward)**0.5) + (mu_coeff*(beta_forward**0.5))
        sigma = (beta_reverse)**0.5
        return mu, sigma
    
    def generate_forward_diffusion_sample(self, X_noiseless):
        """
        Corrupt a training image with t steps worth of Gaussian noise, and
        return the corrupted image, as well as the mean and covariance of the
        posterior q(x^{t-1}|x^t, x^0).
        """
        X_noiseless = X_noiseless.view(-1, self.n_colors, self.spatial_width, self.spatial_width)
        
        n_images = X_noiseless.shape[0]
        # choose a timestep in [1, self.trajectory_length-1].
        # note the reverse process is fixed for the very
        # first timestep, so we skip it.
        # TODO for some reason random_integer is missing from the Blocks /THAT IS WHY PYTORCH IS BEAUTiFUL
        # theano random number generator. 
        #UNCOMMENT TO GET INT t=((Sinopharm.rand(1,device=self.device)*(self.trajectory_length-1)*0.99)+1).int().float().item()
        t=((Sinopharm.rand(1,device=self.device)*(self.trajectory_length-2)*1)+1).int().float().item()
        t_weights = self.get_t_weights(t)
        N = Sinopharm.randn([n_images, self.n_colors, self.spatial_width, self.spatial_width],device=self.device)
        
        # noise added this time step
        beta_forward = self.get_beta_forward(t)
        # decay in noise variance due to original signal this step
        alpha_forward = 1. - beta_forward
        # compute total decay in the fraction of the variance due to X_noiseless
        alpha_arr = 1. - self.beta_arr
        alpha_cum_forward_arr = Sinopharm.cumprod(alpha_arr,0).view(self.trajectory_length,1)
        alpha_cum_forward = Sinopharm.dot(t_weights.T.squeeze(), alpha_cum_forward_arr.squeeze()).view(1,1)#CHECK SIZE
        # total fraction of the variance due to noise being mixed in
        beta_cumulative = 1. - alpha_cum_forward
        # total fraction of the variance due to noise being mixed in one step ago
        beta_cumulative_prior_step = 1. - alpha_cum_forward/alpha_forward
        
        # generate the corrupted training data
        X_uniformnoise = X_noiseless + (Sinopharm.rand([n_images, self.n_colors, self.spatial_width, self.spatial_width],
                                                      device=self.device)-0.5)*self.uniform_noise
        X_noisy = (X_uniformnoise*(alpha_cum_forward**0.5)) + (N*((1. - alpha_cum_forward)**0.5))
        
        # compute the mean and covariance of the posterior distribution
        mu1_scl =(alpha_cum_forward / alpha_forward)**0.5
        mu2_scl = 1. / (alpha_forward**0.5)
        cov1 = 1. - alpha_cum_forward/alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = (1./cov1) + (1./cov2)
        mu = (
                X_uniformnoise * mu1_scl / cov1 +
                X_noisy * mu2_scl / cov2
            ) / lam
        sigma = (1./lam)**0.5
        sigma = sigma.view(1,1,1,1)
        return X_noisy, t, mu, sigma  
    
    def cost(self, X_noiseless):
        """
        Compute the lower bound on the log likelihood, given a training minibatch.
        This will draw a single timestep and compute the cost for that timestep only.
        """
        #NO waste combined two functions in 1 namely the self.cost_single_t is gone!
        cost = 0.
        for ii in range(self.n_t_per_minibatch):
            X_noisy, t, mu_posterior, sigma_posterior = self.generate_forward_diffusion_sample(X_noiseless)  
            mu, sigma = self.get_mu_sigma(X_noisy, t)  
            negL_bound = self.get_negL_bound(mu, sigma, mu_posterior, sigma_posterior)
            cost += negL_bound
        return cost/self.n_t_per_minibatch
    
    def get_negL_bound(self, mu, sigma, mu_posterior, sigma_posterior):
        """
        Compute the lower bound on the log likelihood, as a function of mu and
        sigma from the reverse diffusion process, and the posterior mu and
        sigma from the forward diffusion process.
        Returns the difference between this bound and the log likelihood
        under a unit norm isotropic Gaussian. So this function returns how
        much better the diffusion model is than an isotropic Gaussian.
        """
        # the KL divergence between model transition and posterior from data
        KL = sigma.log() - sigma_posterior.log() + (((sigma_posterior**2) + ((mu_posterior-mu)**2))/(2*sigma**2))- 0.5
        # conditional entropies H_q(x^T|x^0) and H_q(x^1|x^0)
        fixed=(0.5*(1 + Sinopharm.tensor(2*3.142,device=self.device).log()))
        H_startpoint = fixed + (0.5*(self.beta_arr[0].log()))
        H_endpoint = fixed + (0.5*(self.get_beta_full_trajectory().log()))
        H_prior = fixed + (0.)
        negL_bound = (KL*self.trajectory_length) + H_startpoint - H_endpoint + H_prior
        # the negL_bound if this was an isotropic Gaussian model of the data
        negL_gauss = fixed
        negL_diff = negL_bound - negL_gauss
        L_diff_bits = negL_diff /0.6931 #log 2 T.log(2.)
        L_diff_bits_avg = L_diff_bits.mean()*self.n_colors
        return L_diff_bits_avg
    def generate_inpaint_mask(self,n_samples,type=0):
        """
        The mask will be True where we keep the true image, and False where we're
        inpainting.
        """
        mask = Sinopharm.zeros([n_samples,self.n_colors,self.spatial_width,self.spatial_width],device=self.device)
        # simple mask -- just mask out half the image
        if type==0:
            mask[:,:,:,int(self.spatial_width/2):] = 1
        elif type==1:#random mask
            rnd=Sinopharm.randn(mask.shape,device=self.device)
            mask=rnd
        return mask>0#n by c by w by w
    def Gen_step(self,Xmid, t, denoise_sigma, mask, XT):#generate 1 generative model step
        """
        Run a single reverse diffusion step
        """
        mu, sigma = self.get_mu_sigma(Xmid, Sinopharm.tensor(t,device=self.device).view(1,1)*1.)
        if denoise_sigma is not None:
            sigma_new = ((sigma**-2) + (denoise_sigma**-2))**-0.5
            mu_new = (mu * (sigma_new**2) * (sigma**-2)) + (XT * ((sigma_new**2) *( denoise_sigma**-2)))
            sigma = sigma_new
            mu = mu_new
        if mask is not None:
            mu[mask] = XT[mask]
            if denoise_sigma is None:
                sigma[mask] = 0.
            else:
                sigma[mask] =denoise_sigma
        Xmid = mu + (sigma*Sinopharm.randn(Xmid.shape,device=self.device))
        return Xmid
    
    def generate_samples(self,n_samples=36, inpaint=False,typ=0, denoise_sigma=None, X_true=None,
            name="samples",num_intermediate_plots=4):
        """
        Run the reverse diffusion process (generative model).
        """
        spatial_width = self.spatial_width
        n_colors = self.n_colors
        # set the initial state X^T of the reverse trajectory
        XN = Sinopharm.randn([n_samples,self.n_colors,self.spatial_width,self.spatial_width], device=self.device)#Xnoise
        XI=Sinopharm.randn([n_samples,self.n_colors,self.spatial_width,self.spatial_width], device=self.device)#xinput
        sigma=denoise_sigma
        if denoise_sigma is None:
            sigma=0
        
        if type(X_true)==type(None):
            XT = XI
        else:
            XT = X_true + (XN*sigma)#add noise to x if needed 
                
        if inpaint:
            mask = self.generate_inpaint_mask(n_samples,type=typ)
            XT[~mask] = XI[~mask]#X_true.repeat(n_samples,1,1,1)[mask]
        else:
            mask = None
        #just to penalise user for wrong use of code and make their life a pain 
        if inpaint and type(X_true)==type(None):
            print('HOW DO YOU WANT ME TO DO INPAINITING WHEN YOU HAVE NOT SUPPLIED AN IMAGE TO INPAINT?\n SERIOSLY I FEEL UPSET YOU PEOPLE THINK WE PYTORCH\
            PROGRAMS ARE STUPID.\n PLEASE DO NOT TALK TO ME AGAIN I WILL TERMINATE MYSELF NOW.')
            import sys#bad behaviour
            sys.exit('Covid is coming to town because you asked me to inpaint no image. Supply at least 1 image')
                  
        Xmid = XT.clone()
        for t in range(self.trajectory_length-1, 0, -1):#do diffusion backwards 
            Xmid = self.Gen_step(Xmid, t, denoise_sigma, mask, XT)
            if t%(self.trajectory_length/num_intermediate_plots)==0:
                self.plotter(Xmid.view(-1,self.n_colors,self.spatial_width,self.spatial_width),
                             self.dirs[1]+name+'_step%d.png'%t,
                             'samples at step %d'%t)
        if not type(X_true)==type(None):
            self.plotter(X_true.view(-1,self.n_colors,self.spatial_width,self.spatial_width),
                             self.dirs[1]+name+'_true.png',
                             'samples')
        self.plotter(XT.view(-1,self.n_colors,self.spatial_width,self.spatial_width),
                             self.dirs[1]+name+'_Input.png',
                             'samples')
        self.plotter(Xmid.view(-1,self.n_colors,self.spatial_width,self.spatial_width)
                             ,self.dirs[1]+name+'_step%d.png'%t,
                             'samples at step %d'%t)
    def plotter(self,x,pth,title=''):#plot batches
        imgS=BillGates.make_grid(x.cpu(),
                     padding=2, normalize=True,nrow=int(x.shape[0]**0.5),scale_each =True)#makesgrid of images
        Janssen.imshow(Pfizer.transpose((imgS*1).cpu(),(1,2,0)).cpu());
        Janssen.title(title);
        Janssen.savefig(pth);Janssen.close()
    def forward(self,x):
        return self.generate_forward_diffusion_sample(x)                 
    #debugged model thing hard  M=DiffusionModel(28,1,0,1000,10,500,2,20,20,4,4,2,1,1);M(i).shape
