import torch
from quats import rand_quats, outer_prod, approx_rot_dist


#def l1(q1,q2):
#        """ Basic L1 loss """
#        return torch.mean(abs(q1-q2),dim=-1)

#def l2(q1,q2):
#        """ Basic L2 loss """
#        return torch.sqrt(torch.mean((q1-q2)**2,dim=-1))


def l1(q1,q2):
    """ Basic L1 loss """
    return torch.linalg.norm(q1-q2,dim=-1)

def l2(q1,q2):
    """ Basic L2 Loss """
    return torch.linalg.norm(q1-q2,dim=-1)

def mse(q1,q2):
    return torch.sum((q1-q2)**2,dim=-1)


def approx_sqrt(x,a=10):
    eps = 0.5/a
    x_abs = torch.abs(x)
    mask = (x_abs < eps).float()
    x_upper = torch.clamp(x_abs,min=eps)
    x_lower = torch.clamp(x_abs,max=eps)
    output_ts = x_lower*torch.sqrt((a/2)*(1-((a*x_lower)**2)/6))
    output_re = torch.sqrt(torch.log(torch.cosh(a*x_upper))/a)

    output = mask*output_ts + (1-mask)*output_re
    return output


def approx_sqrt_rmse(q1,q2,a=10):
    x = l2(q1,q2)
    return approx_sqrt(x,a)



class Loss:
        """ Wrapper for loss. Inclues option for symmetry as well """
        #def __init__(self,dist_func,syms=None,quat_dim=-1,mean=True):
        def __init__(self,dist_func,syms=None,mean=True):
                self.dist_func = dist_func
                self.syms = syms
                #self.quat_dim = quat_dim
                self.mean = mean
        def __call__(self,q1,q2):
                #q1 = torch.movedim(q1,self.quat_dim,-1)
                #q2 = torch.movedim(q2,self.quat_dim,-1)
                if self.syms is not None:
                        q1_w_syms = outer_prod(q1,self.syms.to(q1.device))
                        if q2 is not None: q2 = q2[...,None,:]
                        dists = self.dist_func(q1_w_syms,q2)
                        dist_min = dists.min(-1)[0]


                        #return torch.mean(dist_min)
                else:
                        dist_min = self.dist_func(q1,q2)

                        #return torch.mean(self.dist_func(q1,q2))
                if self.mean: return torch.mean(dist_min)
                else: return dist_min


        def __str__(self):
                return f'Dist -> dist_func: {self.dist_func}, ' + \
                           f'syms: {self.syms is not None}'


def tanhc(x):
        """
        Computes tanh(x)/x. For x close to 0, the function is defined, but not
        numerically stable. For values less than eps, a taylor series is used.
        """
        eps = 0.05
        mask = (torch.abs(x) < eps).float()
        # clip x values, to plug into tanh(x)/x
        x_clip = torch.clamp(abs(x),min=eps)
        # taylor series evaluation
        output_ts = 1 - (x**2)/3 + 2*(x**4)/15 - 17*(x**6)/315
        # regular function evaluation for tanh(x)/x
        output_ht = torch.tanh(x_clip)/x_clip
        # use taylor series if x is close to 0, otherwise, use tanh(x)/x
        output = mask*output_ts + (1-mask)*output_ht
        return output


def tanh_act(q):
        """ Scale a vector q such that ||q|| = tanh(||q||) """
        return q*tanhc(torch.norm(q,dim=-1,keepdim=True))
        
def safe_divide_act(q,eps=10**-5):
        """ Scale a vector such that ||q|| ~= 1 """
        return q/(eps+torch.norm(q,dim=-1,keepdim=True))




class ActAndLoss(torch.nn.Module):
        """ Wraps together activation and loss """
        def __init__(self,act,loss):
                super(ActAndLoss,self).__init__()
                self.act = act
                self.loss = loss
        def forward(self,X,labels):
                #print(X.shape,labels.shape)
                #X = torch.movedim(X,1,-1)
                #labels = torch.movedim(labels,1,-1)
                #print(X.shape,labels.shape)
                X_act = X if self.act is None else self.act(X)
                return self.loss(X_act,labels)

        #def __call__(self,X,labels):
        #        X_act = X if self.act is None else self.act(X)
        #        return self.loss(X_act,labels)
        def __str__(self):
                return f'Act and Loss: ({self.act},{self.loss})'

        def get_metadata(self):
            return {
                    'syms': self.loss.syms is not None,
                    'loss': self.loss.dist_func.__name__,
                    'act': None if self.act is None else self.act.__name__
                   }


from symmetries import hcp_syms

def get_hcp_losses():

    acts_and_losses = list()
        
    for act in [None,tanh_act,safe_divide_act]:
        for syms in [None,hcp_syms]:
            for dist in [l1,l2,approx_rot_dist]:
                acts_and_losses.append(ActAndLoss(act,Loss(dist,syms)))
    

    for act in [None,tanh_act,safe_divide_act]:
        for syms in [None,hcp_syms]:
            acts_and_losses.append(ActAndLoss(act,Loss(mse,syms)))


    for act in [None,tanh_act,safe_divide_act]:
        for syms in [None,hcp_syms]:
            acts_and_losses.append(ActAndLoss(act,Loss(approx_sqrt_rmse,syms)))



    return acts_and_losses


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        import matplotlib.pyplot as plt

        from symmetries import hcp_syms

        torch.manual_seed(1)
        
        q1 = torch.randn(7,17,19,4)
        q2 = torch.randn(7,17,19,4)

        q1 /= torch.norm(q1,dim=1,keepdim=True)

        q2.requires_grad = True


        #acts_and_losses = list()
        
        #for act in [None,tanh_act,safe_divide_act]:
        #       for syms in [None,hcp_syms]:
        #               for dist in [l1,l2,approx_rot_dist]:
        #                       acts_and_losses.append(ActAndLoss(act,Loss(dist,syms,1)))
        
        acts_and_losses = get_hcp_losses()

        for i,c in enumerate(acts_and_losses):
                print(i,c)
                d = c(q1,q2)
                print(d)

        fig,axes = plt.subplots(2,1)

        for i in range(5):

            x = torch.linspace(-1.2,1.2,301,requires_grad = True)

            if i == 0:
                y = x**2
                label = 'mse'

            if i == 1:
                y = torch.abs(x)
                label = 'rmse'

            if i == 2:
                y = approx_sqrt(x,a=5)
                label = r'approx sqrt $\alpha = 5$'

            if i == 3:
                y = approx_sqrt(x,a=25)
                label = r'approx sqrt $\alpha = 25$'

            if i == 4:
                y = torch.sqrt(torch.abs(x))
                label = 'sqrt'



            l = y.sum()
            l.backward()
            
            axes[0].plot(x.detach(),y.detach(),label=label)
            axes[1].plot(x.detach(),x.grad.detach())

        axes[1].set_ylim(-5,5)

        #x = torch.linspace(-2,2,301,requires_grad = True)
        #y = approx_sqrt(x)


        #l = y.sum()
        #l.backward()

        #import matplotlib.pyplot as plt

        #print(x.grad)

        #plt.plot(x.detach(),y.detach())
        #plt.plot(x.detach(), torch.sqrt(x).detach())
        
        #print(x.grad)

        #plt.plot(x.detach(),x.grad.detach())

        fig.legend()



        plt.show()

    



