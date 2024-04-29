from config import args
from training import *

if __name__ == '__main__':
    if args.network == "WAAE":
        model = WAAE()
        model.train()
