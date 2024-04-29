import argparse
from unityagents import UnityEnvironment
from libs.agents import Agent
from libs.monitor import train, test
from libs import models

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Show pretrained agent in environment", action="store_true")
parser.add_argument("--no_graphics", help="Do not show graphics during training", action="store_true")
parser.add_argument("--environment", nargs='?', help="Pick environment file", default="env/Banana.exe")
parser.add_argument("--checkpoint", nargs='?', help="Pick environment file", default="logs/checkpoint.pth")
parser.add_argument("--model_name", nargs='?', help="Choose a model name. Options: DQN, DuelDQN", default="DQN")
parser.add_argument("--double", help="Enable double DQN", action="store_true")

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
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size    
    state_type = 'discrete'

    # In case state space is continous, get width/height of camara
    if brain.number_visual_observations > 0:
        h = brain.camera_resolutions[0]['height']
        w = brain.camera_resolutions[0]['width']
        state_size = (3, 4, h, w)
        state_type = 'continuous'

    # Get Q networks
    q_local = getattr(models, args.model_name)(state_size, state_type, action_size, random_state=42)
    q_target = getattr(models, args.model_name)(state_size, state_type, action_size, random_state=42)

    # Setup agent
    agent = Agent(
        state_size=state_size,
        state_type=state_type,
        action_size=action_size,
        q_local=q_local,
        q_target=q_target,
        model_name=args.model_name,
        enable_double=args.double,
        random_state=42
    )

    # Testing or training
    if args.test:
        test(env, agent,
            state_type=state_type,
            brain_name=brain_name, 
            checkpoint=args.checkpoint
        )
    else:
        train(env, agent, 
            state_type=state_type,
            brain_name=brain_name, 
            episodes=50000,
            eps_start=1.0, 
            eps_end=0.001, 
            eps_decay=0.97, 
            thr_score=13.0
        )

    # Close environment when done
    env.close()
