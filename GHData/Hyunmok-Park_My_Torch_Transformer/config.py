import argparse

arg_lists = []
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=int, default=1)
parser.add_argument('--num_gpu', type=int, default=0)

parser.add_argument('--num_encoder', type=int, default=4)
parser.add_argument('--num_decoder', type=int, default=4)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_seq', type=int, default=64)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--FFNN_dim', type=int, default=512)

parser.add_argument('--batch_size', type=int, default=2)

def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    return args, unparsed