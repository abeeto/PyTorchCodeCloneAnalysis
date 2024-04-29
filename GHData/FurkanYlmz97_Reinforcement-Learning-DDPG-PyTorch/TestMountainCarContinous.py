from Ddpg import *
import gym

env = gym.make('MountainCarContinuous-v0')

Driver = ActorNN(env.observation_space.shape[0], env.action_space.shape[0], [256, 128])
# LunarLander.load_state_dict(torch.load("ActorEpisode100.pth"))
Driver.load_state_dict(torch.load("CarBestActor.pth"))

noise = OUNoise(env.action_space)
state = env.reset()
noise.reset()

for step in range(1000):
    # state = state + np.random.normal(0, 0.1, 3)  #  Check how robust is your Actor Network
    state = Variable(torch.from_numpy(state.copy()).float())
    action = Driver.forward(state)
    new_state, _, _, _ = env.step(action.detach().numpy())
    env.render()
    state = new_state
env.close()