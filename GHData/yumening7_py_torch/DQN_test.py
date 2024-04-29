
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import gym
import torch.nn as nn

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9                   # Q Learning的策略值，代表90%的情况我们会按照Q表的最优值来选择行为，10%的情况是随机选择行为。
GAMMA = 0.9                     # 奖励衰减值
TARGET_REPLACE_ITER = 100       # 目标更新频率
MEMORY_CAPACITY = 2000          # 记忆库容量
env = gym.make('CartPole-v0')   # 导入gym立杆子实验的模拟场所
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# 接收状态，并输出每种行动的价值
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)        # 随机生成输入初始值
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)        # 随机生成输出初始值

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0                 # for target updating 更新目标步数
        self.memory_counter = 0                     # for storing memory  对应记忆库的位置
        '''
        初始化记忆库，MEMORY_CAPACITY是定义好的记忆库容量。
        这里每一个样本都会存储s(状态)，a(行动)，r(奖励)，s_(下一步状态)
        所以这里的列数应该是N_STATES * 2 （两个s） + 2 （a和r）
        '''
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        # 接下来就是定义optimizer优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        '''
        决策行动
        :param x:
        :return: 决策后的动作action
        '''
        x = torch.unsqueeze(torch.FloatTensor(x), dim=0)  # 增加x的维度
        '''
        np.random.uniform()方法接受三个参数，分别是low(采样下界，默认为0)、high(采样上界，默认为1)、size(输出样本数目，为int或元祖，默认为1)
        所以这里是从[0, 1)之间随机采样
        '''
        if np.random.uniform() < EPSILON:   # 近乎策略值（90%）的可能采取价值最高的动作
            actions_value = self.eval_net.forward(x)
            # 选取action_value中最大的价值
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:           # 随机选取一个动作
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        '''
        存储记忆
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        '''
        # np.hstack代表在水平方向上拼接矩阵
        transition = np.hstack((s, [a, r], s_))
        # 用取余的方式，随着self.memory_counter的递增，可以保证index是取值范围是[0, MEMORY_CAPACITY-1]，下标个数为MEMORY_CAPACITY个
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 每隔100次更新一次target_net, 但eval_net是每一次learn方法都会更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 把eval_net中的所有参数赋值到target_net当中，target_net就实现了更新
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 从记忆库中随机抽取一个批次的下标
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # 去除对应下标的记忆
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # self.eval_net(b_s)得出所有动作的价值，gather(1, b_a)是查找出当初施加动作所对应的价值
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 计算出下一步的q值估计,detach是用于禁止反向传播，因为target_net在上面已经手动更新
        q_next = self.target_net(b_s_).detach()
        # 选出最大的q值估计，并套入Q Learning算法
        q_target= b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # 接着进行误差反向传递
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting exeperience...')

for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_