import os
from datetime import datetime
import warnings

import torch
import wandb
import hydra
from omegaconf import OmegaConf
from torchsummaryX import summary


def extras(cfg):

    cfg.dt_string = datetime.now().strftime("%d:%H:%M:%S")

    print("Disabling python warnings! <config.ignore_warnings=True>")
    warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if not cfg.get("name"):
        print(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py name=experiment_name`"
        )
        print("Exiting...")
        exit()
    
    print(OmegaConf.to_yaml(cfg))

    if not cfg.DEBUG and not os.path.exists(cfg.path.submissions): 
        os.makedirs(cfg.path.submissions)
    save_folder = f'{cfg.name}-{cfg.dt_string}'
    if not cfg.DEBUG and not os.path.exists(os.path.join(cfg.path.weights, save_folder)): 
        os.makedirs(os.path.join(cfg.path.weights, save_folder))


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):

    from src.trainer import Trainer
    from src.utils.utils import save_submission, set_seed, load_dataloader

    set_seed(cfg.seed)
    extras(cfg)

    if cfg.DEBUG:
        cfg.epoch = 1
    else:
        wandb.init(project="dacon-sem", entity="normalkim", config=cfg, name=f'{cfg.name}-{cfg.dt_string}')
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(device)

    # Load data
    train_loader, valid_loader, test_loader = load_dataloader(cfg)
    
    # Init lightning model
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    print(model)
    for data_path, sem, depth in valid_loader:
        break
    summary(model.to(device), sem.to(device), depth.to(device))
    # import pdb; pdb.set_trace()

    trainer = Trainer(cfg, model, device)

    # Train and validation
    print("Start training")
    trainer.fit(train_loader, valid_loader)

    # Test the model
    if not cfg.DEBUG:
        print("Starting testing")
        results = trainer.inference(test_loader)
        save_submission(cfg, results)


if __name__ == "__main__":
    main()