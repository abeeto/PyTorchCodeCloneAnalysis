import argparse
from copy import deepcopy
from pathlib import Path

import gym
import torch

from src.ddpg import DDPG
from src.networks import Actor, Critic
from src.original_networks import (
    Actor as OriginalActor,
    Critic as OriginalCritic,
)
from src.utils import to_tensor_variable
from src.variables import variables


def run(env_str, eval_saved_model, original):
    env = gym.make(env_str)
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]

    models_path = "models/"
    runs_path = "runs/"
    have_models = (
        Path(models_path).exists()
        and len(list(Path(models_path).iterdir())) >= 4
    )
    if have_models:
        print("Loading pre-existent models.")
        actor = torch.load(f"{models_path}actor")
        critic = torch.load(f"{models_path}critic")
        actor_target = torch.load(f"{models_path}actor_target")
        critic_target = torch.load(f"{models_path}critic_target")
    else:
        print("Creating new networks.")
        if original:
            actor = OriginalActor(
                obs_space_size=obs_space_size,
                action_space_size=action_space_size,
            )
            critic = OriginalCritic(
                obs_space_size=obs_space_size,
                action_space_size=action_space_size,
            )
        else:
            actor = Actor(
                obs_space_size=obs_space_size,
                action_space_size=action_space_size,
            )
            critic = Critic(
                obs_space_size=obs_space_size,
                action_space_size=action_space_size,
            )

        actor_target = deepcopy(actor)
        critic_target = deepcopy(critic)

    if not eval_saved_model:
        ddpg = DDPG(
            env=env,
            actor=actor,
            critic=critic,
            actor_target=actor_target,
            critic_target=critic_target,
            gamma=variables["gamma"],
            minibatch_size=variables["minibatch_size"],
            device=variables["device"],
            max_episodes=variables["max_episodes"],
            tau=variables["tau"],
            actor_lr=variables["actor_lr"],
            critic_lr=variables["critic_lr"],
            weight_decay=variables["weight_decay"],
            replay_buffer_size=variables["replay_buffer_size"],
            models_path=models_path,
            runs_path=runs_path,
        )
        ddpg.train()
    else:
        with torch.no_grad():
            observation = env.reset()
            done = False
            while not done:
                env.render()
                action = (
                    actor(to_tensor_variable([observation])).cpu().numpy()[0]
                )
                observation, reward, done, _ = env.step(action)
                if done:
                    print(f"Episode finished with reward {reward}")
                    break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch DDPG")
    parser.add_argument(
        "--env",
        metavar="env",
        dest="env",
        type=str,
        help="""Name of the OpenAI Gym environment.
        Default: LunarLanderContinuous-v2.""",
        default="LunarLanderContinuous-v2",
    )
    parser.add_argument(
        "--eval",
        dest="eval_saved_model",
        action="store_true",
        help="Evaluate on the saved model instead of training.",
    )
    parser.add_argument(
        "--original",
        dest="original",
        action="store_true",
        help="""Use networks as they are defined in the paper, without any optimization.
        Beware that the two types of networks are not interchangeable.""",
    )
    args = parser.parse_args()
    run(args.env, args.eval_saved_model, args.original)
