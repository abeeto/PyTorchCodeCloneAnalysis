
import argparse
parser = argparse.ArgumentParser(
    description='The parameter specificed by operator')


parser.add_argument('--SEED', default=666,
                    type=int, help='the random seed for reproduct the experiment')


# training configurations
parser.add_argument('--BATCH_SIZE', default=32, type=int,
                    help='The batch size in the training phase')
parser.add_argument('--EPISODE', default=1000, type=int,
                    help='How many episodes will the training last for. One TM is loaded in each episode')
parser.add_argument('--MAX_STEP', default=500, type=int,
                    help='How many steps one episode last for')

# running type
parser.add_argument('RUNNING_TYPE', choices=['train', 'retrain', 'test'],
                    type=str, help='How to run models, e.g. retrain, train, or test')

# retrain and loader or checkpoint
parser.add_argument('--OUTPUT_DIR', default='./running_log/model', type=str,
                    help='model salving place')

parser.add_argument('--CHECKPOINT_DIR', default='./running_log/checkpoint',
                    type=str, help='check point path for retrain model')
parser.add_argument('--CHECKPOINT_START_EPISODE', default=10,
                    type=int, help='check point path for retrain model')
parser.add_argument('--CHECK_POINT_INTERVAL', default=10,
                    type=int, help='Save checkpoint every interval')


# priority buffer configuration
parser.add_argument('--MEMORY_SIZE', default=300,
                    type=int, help='prioritized replay buffer parameter memory size')
parser.add_argument('--ALPHA', default=0.6,
                    type=float, help='prioritized replay buffer parameter alpha')
parser.add_argument('--MU', default=0.6,
                    type=float, help='prioritized replay buffer parameter mu')
parser.add_argument('--BETA', default=0.6,
                    type=float, help='prioritized replay buffer parameter beta')
parser.add_argument('--EPSILON_STEPS', default=2700,
                    type=int, help='prioritized replay buffer decay episode step')

parser.add_argument('--cuda', default=0,
                    type=int, help='using cuda')
args = parser.parse_args()

if __name__ == "__main__" or 1:
    print(args)
