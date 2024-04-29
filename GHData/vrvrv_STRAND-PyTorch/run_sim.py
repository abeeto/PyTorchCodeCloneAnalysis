import os
import pickle
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from src import utils
    from src.sim import sim

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # job type is train or test
    true_rank, n, m = config.true_rank, config.n, config.m

    with open(os.path.join(config.data_dir, f"rank_{true_rank}_m_{m}_n_{n}.pkl"), "rb") as f:
        data, params = pickle.load(f)

    with open(os.path.join(config.data_dir, "feature.pkl"), "wb") as f:
        pickle.dump(data['feature'], f)

    with open(os.path.join(config.data_dir, "snv.pkl"), "wb") as f:
        pickle.dump(data['count'], f)

    sim(config)


if __name__ == "__main__":
    main()
