import gym
import numpy as np
from agent import Agent
from utils import label_with_episode_number, plot_learning_curve, save_frames_as_gif

if __name__ == '__main__':
    test = False
    env = gym.make('CartPole-v0')

    
    N = 20 # after 20 steps we learn.
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_games = 250
    
    figure_file = 'plots/cartpole_ppo.png'
    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    frames = []
    
    if test:
        agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            if test:
                # render frames to buffer
                frame = (env.render(mode='rgb_array'))
                frames.append(label_with_episode_number(frame, episode_num=i)) 
            
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            #if test:
            #    env.render()
            if not test:
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score and not test:
            best_score = avg_score
            agent.save_models()
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
    
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(score_history, x, figure_file)        
    env.close()
    save_frames_as_gif(frames, path='./videos', filename='trained_agent.gif')