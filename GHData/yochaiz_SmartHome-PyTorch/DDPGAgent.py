import json
import torch
import torch.backends.cudnn as cudnn


class DDPGAgent:
    mainModelKey = 'main'
    targetModelKey = 'target'
    optimizerKey = 'optimizer'

    def __init__(self, baseFolder, args):
        # load settings JSON
        self.settings = None
        with open(args.settings, 'r') as f:
            self.settings = json.load(f)

        # set CUDA device
        if 'cuda' in self.settings['type']:
            torch.cuda.set_device(args.gpus)
            cudnn.benchmark = True

        # init seeds
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        # init Policy
        policyFname = '{}/{}'.format(baseFolder, self.settings['[policy'])

