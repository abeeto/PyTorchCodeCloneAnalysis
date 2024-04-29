import math
import torch
from torch.optim.optimizer import Optimizer, required

class AdamVanila(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, **kwargs)
        super(AdamVanila, self).__init__(params, defaults)
        
    # momentum2 ( exp_avg_sq ) 에서 적용되는 coefficient 를 구한다.
    def get_momentum2_coeff (self, p, group):
        state = self.state[p]
        beta1, beta2 = group['betas']
        bias_correction2 = 1 - beta2 ** state['step']
        return math.sqrt(bias_correction2)
      
    # 현재 step에서 learning rate 를 구한다. warm-up 이 적용된다.
    def get_lr(self, p, group):
        return group['lr']
    
    # momentum1 ( exp_avg ) 에 적용되는 coefficient 를 구한다.
    def get_momentum1_coeff (self, p, group):
        beta1, beta2 = group['betas']
        return 1.0 / (1 - beta1 ** self.state[p]['step'])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()
                state = self.state[p]
                grad = p.grad.data.float()
                p_data_fp32 = p.data.float()
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                
                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta1).addcmul_(1 - beta1, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
                momentum1_coeff = self.get_momentum1_coeff(p, group)
                momentum2_coeff = self.get_momentum2_coeff(p, group)

                lr = self.get_lr(p, group)
                
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * lr, p_data_fp32)
                if momentum2_coeff != 0:
                    p_data_fp32.addcdiv_(-lr * momentum1_coeff * momentum2_coeff, exp_avg, exp_avg_sq.sqrt().add_(group['eps']))
                else:
                    p_data_fp32.add_(-lr * momentum1_coeff, exp_avg)
                p.data.copy_(p_data_fp32)

class AdamW(AdamVanila):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, warmup = 0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup = warmup)
    
    # warm-up 을 적용한다.
    def get_lr(self, p, group):
        lr = super().get_lr(p, group)
        state = self.state[p]
        if group['warmup'] > state['step']:
            return 1e-8 + state['step'] * lr / group['warmup']
        return lr

class PlainRAdam(AdamVanila):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def get_momentum2_coeff(self, p, group):
        state = self.state[p]
        beta1, beta2 = group['betas']
        beta2_t = beta2 ** state['step']
        N_sma_max = 2 / (1 - beta2) - 1
        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
        beta1, beta2 = group['betas']
        if N_sma >= 5:
            rectified_term = math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2))
            return rectified_term * super().get_momentum2_coeff(p, group)
        else:
            # Variance 발산하면 두번째 momentum은 적용하지 않는다.
            return 0

class RAdam(PlainRAdam):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, buffer_size = 10):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.cache = [(None, None) for _ in range(buffer_size)]
        
    def get_momentum2_coeff(self, p, group):
        state = self.state[p]
        step = state['step']
        cache_index = int(step % len(self.cache))
        saved_step, momentum2_coeff = self.cache[cache_index]
        if saved_step != step:
            momentum2_coeff = super().get_momentum2_coeff(p, group)
            self.cache[cache_index] = (step, momentum2_coeff)
        return momentum2_coeff
