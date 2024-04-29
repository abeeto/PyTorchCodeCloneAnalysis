from env_GoTogether import EnvGoTogether
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.autograd import grad
from torch.distributions.kl import kl_divergence

def flatten(vecs):
    '''
    Return an unrolled, concatenated copy of vecs

    Parameters
    ----------
    vecs : list
        a list of Pytorch Tensor objects

    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    '''

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened

def cg_solver(Avp_fun, b, max_iter=10):
    '''
    Finds an approximate solution to a set of linear equations Ax = b

    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector

    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b

    max_iter : int
        the maximum number of iterations (default is 10)

    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    '''

    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p

def set_params(parameterized_fun, new_params):
    '''
    Set the parameters of parameterized_fun to new_params

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated

    update : torch.FloatTensor
        a flattened version of the parameters to be set
    '''

    n = 0
    for param in parameterized_fun.parameters():
        numel = param.numel()
        new_param = new_params[n:n + numel].view(param.size())
        param.data = new_param
        n += numel

def get_flat_params(parameterized_fun):
    '''
    Get a flattened view of the parameters of a function approximator

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator for which the parameters are to be returned

    Returns
    -------
    flat_params : torch.FloatTensor
        a flattened view of the parameters of parameterized_fun
    '''
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params

def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    '''
    Return a flattened view of the gradients of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated

    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed

    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)

    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself

    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    '''

    if create_graph == True:
        retain_graph = True
    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = torch.cat([v.view(-1) for v in grads])
    return flat_grads

def detach_dist(dist):
    '''
    Return a copy of dist with the distribution parameters detached from the
    computational graph

    Parameters
    ----------
    dist: torch.distributions.distribution.Distribution
        the distribution object for which the detached copy is to be returned

    Returns
    -------
    detached_dist
        the detached distribution
    '''

    detached_dist = Categorical(logits=dist.logits.detach())
    return detached_dist

def mean_kl_first_fixed(dist_1, dist_2):
    '''
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph

    Parameters
    ----------
    dist_1 : torch.distributions.distribution.Distribution
        the first argument to the kl-divergence function (will be fixed)

    dist_2 : torch.distributions.distribution.Distribution
        the second argument to the kl-divergence function (will not be fixed)

    Returns
    -------
    mean_kl : torch.float
        the kl-divergence between dist_1 and dist_2
    '''
    dist_1_detached = detach_dist(dist_1)
    mean_kl = torch.mean(kl_divergence(dist_1_detached, dist_2))
    return mean_kl

def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
    '''
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor (with requires_grad=True)
        the output of the function of which the Hessian is calculated

    inputs : torch.FloatTensor
        the inputs w.r.t. which the Hessian is calculated

    damping_coef : float
        the multiple of the identity matrix to be added to the Hessian
    '''

    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)
    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v
        return Hvp
    return Hvp_fun

class P_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(P_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_score = self.fc3(x)
        return F.softmax(action_score, dim=-1)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class AC():
    def __init__(self, state_dim, action_dim):
        super(AC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_net = P_net(state_dim, action_dim)
        self.q_net = Q_net(state_dim, action_dim)
        self.gamma = 0.99
        self.max_kl = 0.01
        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=1e-2)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.p_net.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def train_TRPO(self, state_list, action_list, prob_list, reward_list, next_state_list):
        state = state_list[0]
        next_state = next_state_list[0]
        for i in range(1, len(state_list)):
            state = np.vstack((state, state_list[i]))
            next_state = np.vstack((next_state, next_state_list[i]))
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        next_a_prob = self.p_net.forward(next_state)
        for epoch in range(5):
            q = self.q_net.forward(state)
            next_q = self.q_net.forward(next_state)
            expect_q = q.clone()
            for i in range(len(state_list)):
                expect_q[i, action_list[i]] = reward_list[i] + self.gamma * torch.sum(next_a_prob[i, :] * next_q[i, :])
            loss = self.loss_fn(q, expect_q.detach())
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()

        q = self.q_net.forward(state)
        a_prob = self.p_net.forward(state)
        v = torch.sum(a_prob * q, 1)
        gae = torch.zeros(len(state_list), )
        for i in range(len(state_list)):
            gae[i] = q[i, action_list[i]] - v[i]

        # train policy
        loss = torch.FloatTensor([0.0])
        for i in range(len(state_list)):
            gae = q[i, action_list[i]] - v[i]
            loss += gae * torch.log(torch.FloatTensor([prob_list[i]]))
        loss = loss / len(state_list)

        g = flat_grad(loss, self.p_net.parameters(), retain_graph=True)

        action_dists = Categorical(probs=a_prob)
        mean_kl = mean_kl_first_fixed(action_dists, action_dists)  # kl-divergence between dist_1 and dist_2
        Fvp_fun = get_Hvp_fun(mean_kl, self.p_net.parameters())
        s = cg_solver(Fvp_fun, g)  # H**-1*g
        betta = torch.sqrt(2 * self.max_kl / torch.sum(g*s))
        current_policy = get_flat_params(self.p_net)
        new_policy = current_policy + betta * s
        set_params(self.p_net, new_policy)

    def load_model(self):
        self.q_net = torch.load('TRPO_q_net.pkl')
        self.p_net = torch.load('TRPO_p_net.pkl')

    def save_model(self):
        torch.save(self.q_net, 'TRPO_q_net.pkl')
        torch.save(self.p_net, 'TRPO_p_net.pkl')

if __name__ == '__main__':
    state_dim = 169
    action_dim = 4
    max_epi = 1000
    max_mc = 500
    epi_iter = 0
    mc_iter = 0
    acc_reward = 0
    reward_curve = []
    env = EnvGoTogether(13)
    agent = AC(state_dim, action_dim)
    for epi_iter in range(max_epi):
        state_list = []
        action_list = []
        prob_list = []
        reward_list = []
        next_state_list = []
        for mc_iter in range(max_mc):
            state = np.zeros((env.map_size, env.map_size))
            state[env.agt1_pos[0], env.agt1_pos[1]] = 1
            state = state.reshape((1, env.map_size * env.map_size))
            action, action_prob = agent.get_action(state)
            group_list = [action, 2]
            reward, done = env.step(group_list)
            next_state = np.zeros((env.map_size, env.map_size))
            next_state[env.agt1_pos[0], env.agt1_pos[1]] = 1
            next_state = next_state.reshape((1, env.map_size * env.map_size,))
            acc_reward += reward
            state_list.append(state)
            action_list.append(action)
            prob_list.append(action_prob)
            reward_list.append(reward)
            next_state_list.append(next_state)
            if done:
                break
        agent.train_TRPO(state_list, action_list, prob_list, reward_list, next_state_list)
        print('epi', epi_iter, 'reward', acc_reward / mc_iter, 'MC', mc_iter)
        env.reset()
        acc_reward = 0