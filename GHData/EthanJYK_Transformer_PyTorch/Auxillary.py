# Optimizer with warm-up stages (Learning rate Scheduler)---------------------#
class LRScheduler:
    def __init__(self, model_size=512, factor=1.0, warm_up=8000, step_from=0):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.997), eps=1e-09)
        self.constant = factor * (model_size ** -0.5)
        self.current_step = step_from
        self.warm_up = warm_up

    def step(self):
        self.current_step += 1
        rate = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def learning_rate(self):
        lr = self.constant
        lr *= min(self.current_step ** (-0.5), self.current_step * self.warm_up ** (-1.5))
        return lr
#-----------------------------------------------------------------------------#



# KL-Divergence Loss with Label Smoothing ------------------------------------#
# y' = (1- epsilon)y + epsilon/K
def LSKLDivLoss(yhat, y, vocab_size, epsilon, padding_idx):
    confidence = 1 - epsilon
    epsilon_K = epsilon / float(vocab_size - 1) # epsilon / K(= vocab - <pad>)
    # create target
    p = torch.zeros_like(yhat).scatter_(1, y.unsqueeze(1), 1) # fill yhat-size 0 matrix with 1s at y indices (dim, index, val)
    # apply smoothing to target
    p = p * confidence + (1 - p) * epsilon_K
    # cross entropy = - p(log_q)
    log_q = F.log_softmax(yhat, dim=-1)
    crossent = -(p * log_q).sum(dim=1)
    crossent = crossent.masked_select(y != padding_idx) # mask paddings out
    # KL Divergence = CrossEntropy - (-p(log_p))
    kldiv = crossent + (confidence * math.log(confidence) + float(vocab_size-1) * (epsilon_K) * math.log(epsilon_K)) # (p_logp (p=1) + p_logp (p=0))
    loss = kldiv.mean()
    return loss
#-----------------------------------------------------------------------------#



# Accuracy -------------------------------------------------------------------#
def get_accuracy(yhat, y, padding_idx):
    yhat = torch.max(yhat, dim=1)[1]
    accurate = torch.eq(yhat, y) # returns 1 for equal indices, 0 for unequal indices
    accurate = accurate.masked_select(y != padding_idx) # returns values at indices where condition == 1
    accuracy = accurate.sum().item()/accurate.size(0)
    return accuracy
#-----------------------------------------------------------------------------#
