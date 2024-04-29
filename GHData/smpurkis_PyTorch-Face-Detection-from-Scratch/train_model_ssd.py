from pathlib import Path

import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace.datamodule_ssd import WIDERFaceDataModuleSSD
from models.ModelMetaSSD import ModelMetaSSD
from models.SSD import SSD

if __name__ == "__main__":
    torch.random.manual_seed(0)

    input_shape = (480, 480)
    filters = 16
    lr = 1e-4

    name = f"ssd_{filters}_{input_shape[0]}x{input_shape[1]}_sam_adam"
    log_path = Path(f"logs/out_{name}.log")
    log_path.unlink(missing_ok=True)
    model_save_path = f"./saved_models/{name}.pth"

    model = SSD(
        filters=filters,
        input_shape=(3, *input_shape),
    ).cuda()

    model.summary(
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        )
    )
    model_setup = ModelMetaSSD(
        model=model,
        lr=lr,
        log_path=log_path,
    )

    # checkpoint = torch.load("lightning_logs/pretrained_mobilenetv3backbone_576_15x15_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt")
    # model_setup.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        gpus=1 if device == torch.device("cuda") else 0,
        max_epochs=70,
        precision=16,
        progress_bar_refresh_rate=1,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )
    dm = WIDERFaceDataModuleSSD(
        batch_size=24,
        input_shape=input_shape,
        shuffle=False,
    )
    trainer.fit(model=model_setup, datamodule=dm)
    model_setup.to_torchscript(model_save_path)
    # trainer.test(model=model_setup, test_dataloaders=dm.test_dataloader())
