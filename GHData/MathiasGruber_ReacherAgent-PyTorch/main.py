import argparse
from unityagents import UnityEnvironment
from libs.agents import Agents
from libs.monitor import train, test


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Show pretrained agent in environment", action="store_true")
parser.add_argument("--no_graphics", help="Do not show graphics during training", action="store_true")
parser.add_argument("--memory", nargs='?', help="Chose memory type", default="replay",  choices=['per', 'replay'])
parser.add_argument("--environment", nargs='?', help="Pick environment file", default="env/Reacher.exe")
parser.add_argument("--checkpoint_actor", nargs='?', help="Pick checkpoint file for actor", default="logs/weights_actor_singleAgent_per.pth")
parser.add_argument("--checkpoint_critic", nargs='?', help="Pick checkpoint file for critic", default="logs/weights_critic_singleAgent_per.pth")


if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()  

    # Start up the environment
    if args.no_graphics:
        env = UnityEnvironment(file_name=args.environment, seed=42, no_graphics=args.no_graphics)
    else:
        env = UnityEnvironment(file_name=args.environment, seed=42)

    # Create environment name based on input file path
    env.name = '_'.join(args.environment.split("/")[:-1])

    # Get dimensions of state space
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size    

    # Number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Setup agent
    agents = Agents(
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
            checkpoint_actor=args.checkpoint_actor,
            checkpoint_critic=args.checkpoint_critic,
            num_agents=num_agents
        )
    else:
        train(env, agents,
            num_agents=num_agents, 
            brain_name=brain_name, 
            n_episodes=50000,             
            thr_score=30.0
        )

    # Close environment when done
    env.close()