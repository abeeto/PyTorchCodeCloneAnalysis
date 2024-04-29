import gym
import numpy as np
from agent import Agent
from environment import Game

if __name__ == "__main__":
    env = Game(human=False, grid=True, infos=False)
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=env.action_space,
        eps_end=0.01,
        input_dims=[env.state_space],
        lr=0.001,
    )

    scores = []
    eps_history = []
    episode = 1

    while env.running:
        score = 0
        done = False
        state = env.reset()

        while not done:
            if not env.running:
                break
            
            env.render()
            action = agent.choose_action(state)
            new_state, reward, done = env.step(action)
            
            score += reward
            agent.store_transitions(state, action, reward, new_state, done)
            agent.learn()

            state = new_state
            scores.append(score)
            eps_history.append(agent.epsilon)


        score = round(score, 1)
        scores.append(score)
        eps = round(agent.epsilon, 2)
        eps_history.append(agent.epsilon)

        avg_score = round(np.mean(scores[-100:]), 1)
        episode += 1

        print(
            f"Episode: {episode}, score: {score}, average score: {avg_score}, epsilon: {eps}, last decision: {agent.last_decision}"
        )
