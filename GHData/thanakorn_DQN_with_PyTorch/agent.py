import torch

class Agent(object):
    def __init__(self, policy, num_actions, device):
        super().__init__()
        self.current_step = 0
        self.policy = policy
        self.num_actions = num_actions
        self.device = device
        
    def act(self, state, policy_net):
        self.current_step += 1
        with torch.no_grad():
            values = policy_net(state).to(self.device)
            action = self.policy.choose_action(values, self.current_step)
            return torch.tensor([action], device=self.device)