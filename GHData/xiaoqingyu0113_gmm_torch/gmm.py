import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch


class GMM_torch:
    def __init__(self, n_components=20, total_iter=30,kmeans_iter=10):
        self.K = n_components
        self.total_iter = total_iter
        self.kmeans_iter = kmeans_iter

    def initialize(self, X,kmeans_iter=10): 
        _, self.N, self.D= X.shape 
        self.device = X.device
        kmeans_clf, kmeans_centroids = kmeans(X, K=self.K, Niter=kmeans_iter)

        # initialize paramters
        self.phi = torch.full([self.K], fill_value=1/(self.K)).to(self.device)
        self.weights = torch.full((self.K,self.N), fill_value=1/(self.K)).to(self.device)
        self.mu = kmeans_centroids.unsqueeze(1)
        self.sigma= bcov(X, aweights=self.weights/self.weights.sum(0))
        self.sigma[torch.isnan(self.sigma)|torch.isinf(self.sigma)]=0

    def e_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(dim=1)

    def m_step(self, X):
        self.mu = (X*self.weights.unsqueeze(2)).sum(dim=1).unsqueeze(1)/self.weights.sum(dim=1).unsqueeze(1).unsqueeze(2) # K1D
        self.mu[torch.isnan(self.mu)|torch.isinf(self.mu)]=0
        self.sigma= bcov(X, aweights=self.weights/self.weights.sum(0))
        self.sigma[torch.isnan(self.sigma)|torch.isinf(self.sigma)]=0

    def fit(self, X):
        X = X.unsqueeze(0)
        self.initialize(X,kmeans_iter=self.kmeans_iter)
        for iter in range(self.total_iter):
            self.e_step(X)
            self.m_step(X)  
            
    def predict_proba(self, X):
        x  =  (X-self.mu).unsqueeze(3)
        numerator =torch.exp(-0.5*(x*torch.transpose(x, 2, 3) * torch.linalg.pinv(self.sigma).unsqueeze(1)).sum((2,3)))
        numerator[torch.isnan(numerator)]=0

        likelihood = numerator/torch.sqrt((2*np.pi)**self.D*torch.linalg.det(self.sigma)).unsqueeze(1)
        likelihood[torch.isinf(likelihood)|torch.isnan(likelihood)]=0

        total_likelihood = likelihood * self.phi.unsqueeze(1)
        weights = total_likelihood / total_likelihood.sum(dim=0).unsqueeze(0)
        weights[torch.isinf(weights)|torch.isnan(weights)]=0

        return weights
    
    def predict(self, X):
        X = X.unsqueeze(0)
        weights = self.predict_proba(X)
        return torch.argmax(weights, axis=0)

    def bic(self):
        raise NotImplementedError

   
    def get_gmm_ellipse(self):
        covariances = self.sigma.cpu()
        means = self.mu.squeeze().cpu()
        self.ellipse_list = []
        vs, ws = torch.linalg.eigh(covariances) #eigenvalues, eigenvectors
        vs = vs.cpu(); ws = ws.cpu()

        for i, (mean, v, w) in enumerate(zip(means, vs,ws)):
            v = 2.0 * np.sqrt(2.0) * torch.sqrt(v)
            u = w[0] / torch.linalg.norm(w[0])
            angle = torch.arctan(u[1] / u[0]) * 180.0  / np.pi  # convert to degree180.0 * angle / np.pi  # convert to degree
            self.ellipse_list.append((i,mean,covariances[i,:,:],angle,u,v))

    def plot_results(self,X,ax=None):
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange",
        "mediumspringgreen", "plum","orangered","green"])
        Y_ = self.predict(X).cpu()
        X = X.cpu()
        self.get_gmm_ellipse()
        if ax == None:
            fig,axes = plt.subplots(1)
        else:
            axes= ax
        for i, ((class_idx, mean,cov,angle,u,v), color) in enumerate(zip(self.ellipse_list, color_iter)):
            axes.scatter(X[Y_==i,0],X[Y_==i,1],0.8,color=color)
            ell = mpl.patches.Ellipse(mean.numpy(), v[0].numpy(), v[1].numpy(), 180.0 + angle.numpy(), color=color)
            ell.set_clip_box(axes.bbox)
            ell.set_alpha(0.3)
            axes.add_artist(ell)
            axes.axis('equal')
            axes.set_xlim(X[:,0].min(),X[:,0].max())
            axes.set_ylim(X[:,1].min(),X[:,1].max())


def kmeans(X, K=15, Niter=10):
    _, N, D = X.shape 
    x = X[0,:,:]
    c = x[(torch.rand(K)*N).to(torch.long), :] 
  
    x_i = x.view(N, 1, D)  
    c_j = c.view(1, K, D)  

    for i in range(Niter):

        D_ij = ((x_i - c_j) ** 2).sum(-1)  
        cl = D_ij.argmin(dim=1).long().view(-1)  

        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl 
        c[torch.isinf(c)|torch.isnan(c)]=x.max()/2
    return cl, c

def bcov(points,aweights=None):
    K, N, D = points.size()
    if aweights != None:
        K,_ = aweights.size()
    if aweights == None:
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(K * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(K, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    else:
        aweights = aweights*(N-1)
        mean = (points*aweights.unsqueeze(2)).sum(dim=1).unsqueeze(1) / aweights.sum(dim=1).unsqueeze(1).unsqueeze(2) # K1D
        diffs = (points - mean).reshape(K * N, D) #
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(K, N, D, D)
        aweights = aweights.unsqueeze(2).unsqueeze(3)
        bcov = (prods*aweights).sum(dim=1) / (aweights.sum(dim=1)-1)
    return bcov   

