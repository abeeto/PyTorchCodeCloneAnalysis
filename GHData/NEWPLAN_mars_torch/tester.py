

class Tester():
    def __init__(self, agent, env, model_path, num_episodes=10, max_ep_steps=100, test_ep_steps=100):
        self.EPISODE = num_episodes
        self.MAX_STEP = max_ep_steps
        self.agent = agent
        self.env = env
        self.agent.IS_TRAINING = False
        self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)
        pass

    def test(self, debug=False, visualize=True):
        avg_reward = 0
        for episode in range(self.EPISODE):
            # reset at the start of episode
            s0 = self.env.reset()
            episode_steps = 0
            episode_reward = 0.

            # start episode
            done = False
            for step in range(self.MAX_STEP):

                if visualize:
                    self.env.render()

                action = self.policy(s0)

                s0, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' %
                      (episode, episode_reward))

            avg_reward += episode_reward

        avg_reward /= self.EPISODE

        print("avg reward: %5f" % (avg_reward))
