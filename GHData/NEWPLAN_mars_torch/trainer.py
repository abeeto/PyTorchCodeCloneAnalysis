

from log import logger


class Trainer():
    def __init__(self, agent, env, config, record=True):
        self.env = env
        self.agent = agent
        self.config = config['args']
        pass

    def train(self, start_episode=0, start_step=0):

        for episode in range(start_episode, self.config.EPISODE):

            u_max, tunnel_util, s0 = self.env.reset(
                episode)  # back up for network env
            episode_reward = 0
            logger.info("Max util: {}".format(u_max))

            for step in range(start_step, self.config.MAX_STEP):
                # env.render()
                a0 = self.agent.act(s0)
                # print(a0)

                u_max, tunnel_util, link_util = self.env.step(a0)
                s1, r1 = link_util, u_max*-0.5
                self.agent.store_transaction(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self.agent.learn()
                if step % 20 == 0:
                    logger.info("{}:{}, maximum utilization: {},reward:{}".format(
                        episode, step, u_max, r1))
            start_step = 0

            # TODO: save checkpoint
            if self.config.CHECK_POINT_INTERVAL > 0 and episode % self.config.CHECK_POINT_INTERVAL == 0:
                print("Saving checkpoint at episode:", episode)
                self.agent.save_checkpoint(
                    episode, step, self.config.CHECKPOINT_DIR)

            print(episode, ': ', episode_reward)
        self.agent.save_model(self.config.OUTPUT_DIR)
