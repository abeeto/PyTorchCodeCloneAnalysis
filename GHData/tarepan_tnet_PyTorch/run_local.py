from pathlib import Path

from StarGAN_VC.main import main
from mnet.getConfigs import getConfigs

class Dirs:
    def __init__(self, models, trains, stats, evals, log, stores):
        self.models = models
        self.trains = trains # MCEPseqs
        self.stats = stats # stats
        self.evals = evals # wavs
        self.log = log
        self.stores = stores

if __name__ == "__main__":
    root = Path(".")
    dirs = Dirs(
        root/"trials"/"trial1"/"models", # model
        root/"processed_data"/"trains",  # MCEPseq
        root/"processed_data"/"commons", # stats
        root/"processed_data"/"evals",   # wavs
        root/"log", # log
        root/"trials"/"trial1"/"generated" # stores
    )
    args = getConfigs("f")
    # args.batch_num_limit_train = 2
    # args.batch_num_limit_test = 2
    args.log_interval = 1
    args.cls_dim = 4

    main(args, dirs)
