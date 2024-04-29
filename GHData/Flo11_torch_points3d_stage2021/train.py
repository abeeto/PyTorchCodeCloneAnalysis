import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os
from torch_points3d.trainer import Trainer

import mlflow
import mlflow.sklearn

@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    # mlflow settings
    mlflow.set_tracking_uri(hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.job_name)

    with mlflow.start_run():
        # mlflow basics parameters
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_param("epochs", cfg.epochs)
        mlflow.log_param("main_sampler", cfg.data.main_sampler)
        mlflow.log_param("first_subsampling", cfg.data.first_subsampling)
        mlflow.log_param("grid_param", cfg.data.grid_param)
        mlflow.log_param("radius_param", cfg.data.grid_param)

        trainer = Trainer(cfg)
        trainer.train()

    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
