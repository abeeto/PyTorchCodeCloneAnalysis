import argparse
from unityagents import UnityEnvironment
from libs.agents import MADDPG
from libs.monitor import train, test


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Show pretrained agent in environment", action="store_true")
parser.add_argument("--no_graphics", help="Do not show graphics during training", action="store_true")
parser.add_argument("--memory", nargs='?', help="Chose memory type", default="replay",  choices=['per', 'replay'])
parser.add_argument("--environment", nargs='?', help="Pick environment file", default="env/Tennis.exe")
parser.add_argument("--name", nargs='?', help="Pick name for this training", default="tennis")


if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()  

    # Start up the environment
    if args.no_graphics:
        env = UnityEnvironment(file_name=args.environment, seed=42, no_graphics=args.no_graphics)
    else:
        env = UnityEnvironment(file_name=args.environment, seed=42)

    # Create environment name based on input file path
    env.name = args.name

    # Get dimensions of state space
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    # Number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Setup agent
    agents = MADDPG(
        state_size=state_size,
        action_size=action_size,
        num_agents=num_agents,
        memory=args.memory,
        random_state=42
    )

    # Testing or training
    if args.test:
        test(env, agents,
            brain_name=brain_name, 
            num_agents=num_agents
        )
    else:
        train(env, agents,
            num_agents=num_agents, 
            brain_name=brain_name, 
            n_episodes=10000,             
            thr_score=0.5
        )

    # Close environment when done
    env.close()