import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random as np
import time
# import networks
from Network.actor import Actor, ActorNet
from Network.critic import Critic, CriticNet

from ReplayBuffer.replaybuffer import PriorityBuffer
from Explorer.explorer import MyExplorer

import os

from log import logger

# DDPGAgent


class DDPGAgent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.IS_TRAINING = True

        print("Using cuda for training? ", self.using_cuda)

        s_dim = self.env.get_state_dim()
        a_dim = self.env.get_action_dim()

        self.beta_ = self.args.BETA

        print("\nShowing the env information:\nState dimension:{}, Action dimension:{}\n".format(
            s_dim, a_dim))

        self.explorer_ = MyExplorer(a_dim, 0.1)

        self.actor = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)

        self.critic = CriticNet(s_dim, sum(a_dim))

        self.critic_target = CriticNet(s_dim, sum(a_dim))
        self.actor_optim = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

        # Create priority replay buffer:
        self.buffer = []
        # self.replaybuffer_ = PriorityBuffer(
        #     self.args.MEMORY_SIZE,
        #     self.args.BATCH_SIZE,
        #     self.args.ALPHA,
        #     self.args.MU,
        #     self.args.SEED)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.using_cuda:
            logger.info("Loading to GPU space")
            self.cuda()

    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        if self.using_cuda:
            s0 = s0.cuda()
        a0 = self.actor(s0).squeeze(0).detach().cpu().numpy()
        return self.explorer_.get_action(a0)
        return a0

    def store_transaction(self, *transition, td_error=0, gradient=0):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        return

        # self.replaybuffer_.add(transition, td_error, [
        #                        np.mean(np.abs(gradient))])

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        #sta1 = time.time()

        # new added
        #self.beta_ += (1 - self.beta_) / self.args.EPSILON_STEPS
        #batch_, weights_, indices_ = self.replaybuffer_.select(self.beta_)

        #weights_ = np.expand_dims(weights_, axis=1)
        # new added
        samples = np.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)
        #sta2 = time.time()
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)
        #sta3 = time.time()
        if self.using_cuda:
            s0 = s0.cuda()
            a0 = a0.cuda()
            r1 = r1.cuda()
            s1 = s1.cuda()
        #sta4 = time.time()
        #print("time cost: ", (sta2-sta1)*1000, sta3-sta3, (sta4-sta3)*1000)

        def critic_learn(is_weight=None):
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            # loss=torch.multiply(loss,is_weight)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():  # compute grad in one stage
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def actor_learn_two_stage():
            actor_out = self.actor(s0)  # temp storage
            actor_out1 = actor_out.detach()  # reconstruct input of critic
            actor_out1.requires_grad = True
            loss = -torch.mean(self.critic(s0, actor_out1))
            self.actor_optim.zero_grad()
            loss.backward()  # derive dQ/da
            actor_out.backward(actor_out1.grad)  # derive: dQ/dθ= dQ/da * da/dθ
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau)
        #sta5 = time.time()
        critic_learn()
        #sta6 = time.time()
        actor_learn()
        # actor_learn_two_stage()
        #sta7 = time.time()
        # self.replaybuffer_.priority_update(indices, np.array(
        #    batch_error).flatten(), np.mean(np.abs(grads[0]), axis=1))
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        #sta8 = time.time()

        # print("time cost: ", (sta6-sta5)*1000,
        #      (sta7-sta6)*1000, (sta8-sta7)*1000)

    def load_weights(self, output):  # load parameters from output file
        if output is None:
            print("load weight failed")
            return
        print('Loading policy networks')
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

        print('Loading target networks')
        self.actor_target.load_state_dict(
            torch.load('{}/actor.pkl'.format(output)))
        self.critic_target.load_state_dict(
            torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        if not os.path.exists(output):
            os.makedirs(output)
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
        print('Saving models in ', output)

    def save_config(self, output, save_obj=False):

        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

        if save_obj:
            file = open(output + '/config.obj', 'wb')
            pickle.dump(self.config, file)
            file.close()

    def save_checkpoint(self, ep, step, output):

        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)

        torch.save({
            'episode': ep,
            'step': step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, '%s/checkpoint_ep_%d.tar' % (checkpath, ep))

    def load_checkpoint(self, model_path, pointer=0):
        model_path = model_path + \
            '/checkpoint_model/checkpoint_ep_' + str(pointer) + '.tar'
        print("reloading model from ", model_path)
        checkpoint = torch.load(model_path)
        episode = checkpoint['episode']
        step = checkpoint['step']
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

        return episode, step

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def show_model(self):
        logger.info("Showing actor model: ")
        for item in self.actor.named_parameters():
            print(item[0], item[1].shape)

        logger.info("Showing critic model: ")
        for item in self.critic.named_parameters():
            print(item[0], item[1].shape)

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        print("Streaming:::::::::", stream1, stream2)
