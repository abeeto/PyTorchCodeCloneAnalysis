import torch
import numpy as np
import math


class OHEM():
    def __init__(self, exp=None, groups=1, init_prop=1.0, final_prop=0.15, steps=1e8, verbose=False):
        if verbose:
            #print('OHEMLoss reduction_dim:', reduction_dim)
            print('OHEMLoss exp:', exp)
            print('OHEMLoss groups:', groups)
            print('OHEMLoss init_prop:', init_prop)
            print('OHEMLoss final_prop:', final_prop)
            print('OHEMLoss steps:', steps)
        #self.reduction_dim = reduction_dim
        self.exp = exp
        self.curr_exp = 0.
        self.groups = groups
        self.init_prop = min(init_prop, 1)
        self.final_prop = max(final_prop, 0)
        self.steps = steps
        self.curr_prop = init_prop
        self.curr_step = 0
    

    def __call__(self, loss, reduction_dim=None):
        if self.exp is not None:
            if reduction_dim is not None:
                loss = torch.mean(loss, reduction_dim)
            numel = loss.numel()
            weight = torch.ones(numel).to(loss.get_device())
            for i in range(numel):
                weight[i] = ((numel - i)/numel) ** float(self.curr_exp)
            sorted_loss, _ = torch.sort(loss.flatten(), descending=True, stable=False)
            return (sorted_loss * weight).mean() * (numel / weight.sum())
        elif self.groups > 1:
            if reduction_dim is not None:
                loss = torch.mean(loss, reduction_dim)
            
            props = []#[i for i in np.arange(1/self.groups, 1, 1/self.groups)]
            for i in range(1, self.groups):
                props.append(1 / (2**i))
            #props.append(1.0)
            sorted_loss, _ = torch.sort(loss.flatten(), descending=True, stable=False)
            total_loss = loss.mean()
            for prop in props:
                topn = math.ceil(loss.numel() * prop)
                total_loss += sorted_loss[:topn].mean()
            return total_loss / self.groups
        else:
            # Skip
            if self.curr_prop == 1:
                return loss.mean()
            
            if reduction_dim is not None:
                loss = torch.mean(loss, reduction_dim)
            
            topn = math.ceil(loss.numel() * self.curr_prop)
            val, _ = torch.topk(loss.flatten(), topn)
            return val.mean()
    

    def step(self):
        self.curr_step += 1
        if self.init_prop != self.final_prop:
            self.curr_prop = self.final_prop + (self.init_prop - self.final_prop) * math.cos(min(self.curr_step / self.steps, 1)*(math.pi / 2))
        self.curr_exp = self.exp * min(self.curr_step / self.steps, 1)
        print('OHEM: curr_step = %d/%d' % (self.curr_step, self.steps))
        if self.exp is not None:
            print('OHEM: curr_exp =', self.curr_exp)
        elif self.groups > 1:
            print('OHEM: groups =', self.groups)
        else:
            print('OHEM: curr_prop =', self.curr_prop)
        


class OHEMLoss(OHEM):
    def __init__(self, criterion, reduction_dim=None, exp=None, groups=1, init_prop=1.0, final_prop=0.15, steps=1e8, verbose=False):
        
        self.criterion = criterion
        self.reduction_dim = reduction_dim
        super(OHEMLoss, self).__init__(exp=exp, groups=groups, init_prop=init_prop, final_prop=final_prop, steps=steps, verbose=verbose)
        if verbose:
            print('OHEMLoss criterion:', self.criterion)
    

    def __call__(self, input, target):
        loss = self.criterion(input, target)
        return super(OHEMLoss, self).__call__(loss, reduction_dim=self.reduction_dim)
    

def test():
    from IoULoss import IoULoss
    steps = 50
    dummy_input = torch.randn(2, 6, 256, 256)
    dummy_target = torch.randn(2, 6, 256, 256)
    dummy_criterion = IoULoss(pos_weight=torch.ones(6), neg_weight=torch.ones(6), reduction='none')
    ohem = OHEMLoss(dummy_criterion, init_prop=1.0, final_prop=0.15, steps=steps, verbose=True)
    loss = ohem(dummy_input, dummy_target)
    print(loss)
    for i in range(steps+10):
        ohem.step()


if __name__ == '__main__':
    test()
