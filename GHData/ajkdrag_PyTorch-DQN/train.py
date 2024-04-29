import random
import numpy as np
from collections import deque

from network.mlp import MLP
from model.dqn import DQN
from utils.general import get_device
from utils.visualize import plot_results


class ReplayBuffer:
    def __init__(self, num_categories, maxlen) -> None:
        self.num_categories = num_categories
        self.maxlen = maxlen
        self.len = 0
        for i in range(num_categories):
            setattr(self, f"buffer_{i}", deque(maxlen=maxlen))

    def __len__(self):
        return self.len

    def append(self, *args):
        assert len(args) == self.num_categories
        for i, val in enumerate(args):
            buffer = getattr(self, f"buffer_{i}")
            buffer.append(val)
        self.len = min(self.len + 1, self.maxlen)

    def sample(self, sample_size):
        return list(
            zip(
                *random.sample(
                    list(
                        zip(
                            *[
                                getattr(self, f"buffer_{i}")
                                for i in range(self.num_categories)
                            ]
                        )
                    ),
                    sample_size,
                )
            )
        )


class Trainer:
    def __init__(self, env, opts, hyps):
        self.env = env
        self.opts = opts
        self.hyps = hyps
        # extract vars
        self.observation_space = len(self.env.observation_space)
        self.action_space = self.env.action_space.n
        self.exploration_list = []
        self.score_list = []

        self.batch_sz = self.hyps["batch_sz"]
        self.device = get_device()

    def setup_model(self):
        policy_net = MLP(self.observation_space, self.action_space)
        target_net = MLP(self.observation_space, self.action_space)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        self.model = DQN(policy_net, target_net, self.device, self.hyps)
        self.model.setup()
        print(self.model.policy_net)

    def setup_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            5, maxlen=self.hyps["mem_sz"]
        )  # s, a, r, ns, d

    def setup(self):
        self.setup_replay_buffer()
        self.setup_model()

    def stop_running(self):
        return all(np.array(self.score_list[-4:]) >= self.hyps["tgt_reward"])

    def run(self):
        run = 0
        all_steps = 0

        running = True
        try:
            while running:
                run += 1
                state = self.env.reset()
                state = np.reshape(state, (1, self.observation_space))

                step = 0
                rewards = 0

                while True:
                    step += 1
                    all_steps += 1

                    if (
                        self.opts["display"]
                        # and len(self.replay_buffer) == self.hyps["mem_sz"]
                        and run >= 200
                    ):
                        self.env.render()

                    action = self.model.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    rewards += reward

                    # reward = reward if not done else -reward
                    next_state = np.reshape(next_state, (1, self.observation_space))
                    self.replay_buffer.append(state, action, reward, next_state, done)
                    state = next_state

                    if done:
                        print(
                            "run",
                            run,
                            "exploration",
                            self.model.exploration_rate,
                            "score",
                            rewards,
                        )

                        self.exploration_list.append(self.model.exploration_rate)
                        # self.score_list.append(step)
                        self.score_list.append(rewards)

                        if self.stop_running():
                            running = False

                        break

                if len(self.replay_buffer) < self.hyps["mem_sz"]:
                    continue

                batch = self.replay_buffer.sample(self.batch_sz)
                self.model.fit_one_cycle(batch)

                if all_steps % self.hyps["sync_tgt_steps"] == 0:
                    self.model.target_net.load_state_dict(
                        self.model.policy_net.state_dict()
                    )
        except KeyboardInterrupt:
            return
        finally:
            # post train stuff
            self.model.save(self.opts["model_dir"])
            plot_results(self.exploration_list, self.score_list, self.opts["model_dir"])

