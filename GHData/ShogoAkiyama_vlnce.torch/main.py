import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.config.default import get_config


def main():
    exp_config = './vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml'
    opts = None
    config = get_config(exp_config, opts)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    # trainer.train()
    trainer.inference()


if __name__ == '__main__':
    main()