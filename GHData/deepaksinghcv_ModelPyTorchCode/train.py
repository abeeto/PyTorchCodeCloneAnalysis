import argparse
from codebase.config.configReader import *
from codebase.logger.logger import *

def trainer(cfg,args,logger):
    pass
    #get the model

    #get the optimizer

    #get the loss function

    #train the model
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "trainer file argument description")

    parser.add_argument(
        "--config",
        type = str,
        help = "location of the yaml configuration file."
    )

    parser.add_argument(
        "--model_save_location",
        type = str,
        default = "./",
        help = "location to save the model"
    )

    args = parser.parse_args()

    config = ConfigReader(args.config)
    cfg = config.getConfig()

    logCfg = cfg["logging"]
    log_root_obj = CustomLogger(logCfg)
    logger = log_root_obj.getLogger()

    trainer(cfg,args,logger)
