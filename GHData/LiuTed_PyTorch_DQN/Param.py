import gym
import torch
from torchvision import transforms as transf
import numpy as np

BATCH_SIZE = 32
LEARNING_RATE = 3e-3
GAMMA = 0.9
MEMORY_SIZE = 100000
UPDATE = 20
IMG_SIZE = [72, 72]
NUM_EPISODE = 100
STEP_PER_EPISODE = 10000
DEBUG = True
TEST_EPISODE = 100
DO_TEST_EVERY_LOOP = 20

def get_env(name = 'CartPole-v0'):
    env = gym.make(name)
    env.reset()
    return env

if torch.cuda.is_available() and not DEBUG:
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

def get_screen(env):
    screen = env.render(mode='rgb_array')
    screen = screen.transpose((2, 0, 1)) # (HWC)->(CHW)
    screen = torch.from_numpy(screen.astype(np.float32) / 255)
    screen = transf.ToPILImage()(screen)
    screen = transf.Resize(IMG_SIZE)(screen)
    screen = transf.ToTensor()(screen).view(3, IMG_SIZE[0], IMG_SIZE[1])
    return screen
