from Ddpg import *
import gym

env = gym.make('LunarLanderContinuous-v2')

LunarLander = ActorNN(env.observation_space.shape[0], env.action_space.shape[0], [256, 128])
# LunarLander.load_state_dict(torch.load("ActorEpisode100.pth"))
LunarLander.load_state_dict(torch.load("LunarBestActor.pth"))

noise = OUNoise(env.action_space)
state = env.reset()
noise.reset()

for step in range(1000):
    # state = state + np.random.normal(0, 0.1, 3)  #  Check how robust is your Actor Network
    state = Variable(torch.from_numpy(state.copy()).float())
    action = LunarLander.forward(state)
    new_state, _, _, _ = env.step(action.detach().numpy())
    env.render()
    state = new_state
env.close()
