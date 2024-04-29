from hydra import compose, initialize

from cutpaste.base_trainer import BaseTrainer


def main() -> None:
    initialize(version_base=None, config_path="./conf", job_name="cut_paste")
    cfg = compose(config_name="train_config")
    trainer = BaseTrainer(cfg)
    trainer.model_config()
    trainer.train()


if __name__ == "__main__":
    main()
