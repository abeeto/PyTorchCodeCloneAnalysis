import Net
import Memory
import Param
import random
import torch

class DQN(object):
    def __init__(self, capacity, env, eps_sche):
        self.memory = Memory.Memory(capacity)
        self.eps_scheduler = eps_sche

        screen = Param.get_screen(env)
        _, h, w = screen.shape
        self.num_act = env.action_space.n
        # self.policy = Net.FCN(h, w, self.num_act).to(Param.device)
        # self.target = Net.FCN(h, w, self.num_act).to(Param.device)
        # self.state_based = False
        self.policy = Net.FullyConnected(env.observation_space.shape[0], self.num_act).to(Param.device)
        self.target = Net.FullyConnected(env.observation_space.shape[0], self.num_act).to(Param.device)
        self.state_based = True
        
        self.target.load_state_dict(self.policy.state_dict())
        self.target.train(False)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=Param.LEARNING_RATE,
            weight_decay=0.001,
            amsgrad=True
        )
        self.step = 0

    def get_action(self, state, training=True):
        r = random.random()
        eps = self.eps_scheduler.eps
        if(r < eps and training):
            return torch.tensor([random.randrange(self.num_act)], device=Param.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy(state.to(Param.device)).max(1)[1]
    
    def push(self, val): # val should be [state, action, next_state, reward]
        self.memory.push(val)
    
    def learn(self):
        if(len(self.memory) < Param.BATCH_SIZE):
            return
        batch = self.memory.sample(Param.BATCH_SIZE)

        if self.state_based:
            states = torch.tensor([v[0] for v in batch], dtype=torch.float32, device=Param.device)
            next_states = torch.tensor([v[2] for v in batch if v[2] is not None], dtype=torch.float32, device=Param.device)
        else:
            states = torch.stack([v[0] for v in batch], 0).to(Param.device)
            next_states = torch.stack([v[2] for v in batch if v[2] is not None], 0).to(Param.device)

        actions = torch.tensor([[v[1]] for v in batch], dtype=torch.long, device=Param.device)

        #version 1
        # if_nonterm_mask = [[v[2] is not None] for v in batch]
        # if_nonterm_mask = torch.tensor(if_nonterm_mask, dtype=torch.bool, device=Param.device)
        # rewards = torch.tensor([[v[3]] for v in batch], dtype=torch.float32, device=Param.device)

        # next_action_values = torch.zeros_like(rewards).to(Param.device)
        # next_action_values[if_nonterm_mask] = self.target(next_states).max(1)[0]

        # y = rewards + next_action_values * Param.GAMMA
        
        #version 2
        with torch.no_grad():
            y = [[v[3] if v[2] is None else (v[3]+Param.GAMMA * self.target(torch.tensor([v[2]], dtype=torch.float32)).detach().max(1)[0])] for v in batch]
            y = torch.tensor(y, dtype=torch.float32)

        self.optimizer.zero_grad()
        Q_sa = self.policy(states).gather(1, actions)
        loss = torch.nn.functional.l1_loss(Q_sa, y)
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if(self.step % Param.UPDATE == 0):
            self.target.load_state_dict(self.policy.state_dict())
        self.eps_scheduler.update()
        return loss
    
    def save(self, pos):
        torch.save(self.policy.state_dict(), pos)
    
    def restore(self, pos):
        self.policy.load_state_dict(torch.load(pos))
