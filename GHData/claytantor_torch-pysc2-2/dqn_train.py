import sys
import os
import datetime
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS

from deepq import dqn

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
max_mean_reward = 0
last_filename = ""


def main():
    FLAGS(sys.argv)

    with sc2_env.SC2Env(
            map_name="DefeatRoaches",
            players=[sc2_env.Agent(sc2_env.Race.terran, "Tergot"),
                     sc2_env.Bot(sc2_env.Race.random,
                                 sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=86, minimap=64),
                use_feature_units=True),
            step_mul=4,
            realtime=False,
            save_replay_episodes=0,
            visualize=False) as env:

        dqn.learn(
            env,
            num_actions=3,
            lr=1e-4,
            max_timesteps=10000000,
            buffer_size=100000,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            train_freq=2,
            learning_starts=100000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=True,
            num_cpu=2
        )





if __name__ == '__main__':
    main()
