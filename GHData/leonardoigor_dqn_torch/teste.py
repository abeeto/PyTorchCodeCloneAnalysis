import numpy as np
import torch


def test(env, agent, test_aps):
    for ep_cnt in range(test_aps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
        print("Episode: {}/{} | Reward: {}".format(ep_cnt, test_aps, ep_reward))


def fill_memory(env, agent, memory_fill_eps):
    for _ in range(memory_fill_eps):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            state = next_state


def train(env, agent, train_eps, memory_fill_eps, batch_size, update_freq, model_filename):

    fill_memory(env, agent, memory_fill_eps)
    print("Memory filled")
    print("len(replay_memory) = ", len(agent.replay_memory))

    step_cnt = 0
    reward_history = []
    best_score = -np.inf

    for eps in range(train_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            agent.learn(batch_size)

            if step_cnt % update_freq == 0:
                agent.update_target()

            state = next_state
            ep_reward += reward
            step_cnt += 1
        agent.update_target()
        reward_history.append(ep_reward)
        current_avg_score = np.mean(reward_history[-100:])
        print("Episode: {}/{} | Steps: {} | Reward: {} | Avg reward: {}".format(
            eps, train_eps, step_cnt, ep_reward, current_avg_score))

        if current_avg_score >= best_score:
            agent.save(model_filename)
            best_score = current_avg_score
