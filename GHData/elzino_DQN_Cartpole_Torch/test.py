import gym
from itertools import count

from agent import *
from env import *
from arguments import *


def main_cnn(load_path, num_episode):
    load_path = str(load_path)
    num_episode = int(num_episode)

    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    policy_net = CnnQLearning(screen_height, screen_width).to(device)
    policy_net.load_state_dict(torch.load(load_path))

    steps_done = 0
    rewards = []

    for i_episode in range(num_episode):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, steps_done, eps_start=0, eps_end=0, eps_decay=0)
            _, _, done, _ = env.step(action.item())  # TODO env step github code 보기

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)

            if t >= 5000:
                done = True

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Move to the next state
            state = next_state
            steps_done += 1

            if done:
                print('episode: {} --- reward: {}'.format(i_episode, t))
                rewards.append(t)
                break

    average = sum(rewards) / float(len(rewards))
    print('average reward: {}'.format(average))
    print('Complete')
    env.render()
    env.close()


def main_dnn(load_path, num_episode):
    load_path = str(load_path)
    num_episode = int(num_episode)

    env = gym.make('CartPole-v0').unwrapped

    policy_net = DnnQLearning().to(device)
    policy_net.load_state_dict(torch.load(load_path))

    steps_done = 0
    rewards = []

    for i_episode in range(num_episode):
        state = env.reset()
        state = torch.tensor([state], dtype=torch.float32, device=device)
        for t in count():
            env.render()
            # Select and perform an action
            action = select_action(policy_net, state, steps_done, eps_start=0, eps_end=0, eps_decay=0)
            next_state, _, done, _ = env.step(action.item())  # TODO env step github code 보기
            state = torch.tensor([next_state], dtype=torch.float32, device=device)

            steps_done += 1

            if t >= 5000:
                done = True

            if done:
                print('episode: {} --- reward: {}'.format(i_episode, t))
                rewards.append(t)
                break

    average = sum(rewards) / float(len(rewards))
    print('average reward: {}'.format(average))
    print('Complete')
    env.render()
    env.close()


if __name__ == '__main__':
    args = vars(get_args_test())
    if args['mode'] == 'dnn':
        args.pop('mode')
        main_dnn(**args)
    elif args['mode'] == 'cnn':
        args.pop('mode')
        main_cnn(**args)
    else:
        print('choose mode among cnn and dnn')
