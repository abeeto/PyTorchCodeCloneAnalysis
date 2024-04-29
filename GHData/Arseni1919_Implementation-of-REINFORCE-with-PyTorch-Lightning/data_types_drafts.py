from IMPORTS import *


# Named tuple for storing experience steps gathered in training
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):  # -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self): # -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


replay_buffer = ReplayBuffer(1000)
dataset = RLDataset(replay_buffer, sample_size=12)
dataloader = DataLoader(dataset=dataset)

env = gym.make("CartPole-v0")
state = env.reset()
for i in range(1000):
    # do step in the environment
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    exp = Experience(state, action, reward, done, new_state)
    replay_buffer.append(exp)
    state = new_state
    if done:
        state = env.reset()

# print(len(dataloader))

for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched['image'].size(),
    #       sample_batched['landmarks'].size())
    print(f'num of batch = {i_batch}')
    print(f'len of sample in batch = {len(sample_batched[0])}')
    print('-------------------')
    # print(i_batch, sample_batched)
    print('-------------------')
    print('-------------------')
    # if i_batch > 20:
    #     break

# for i in range(3):
#     sample = my_dataset[i]
#     print(i, torch.tensor(sample))
#     print('-------------------')
#     print(i, sample)
#     print('-------------------')
#     print('-------------------')