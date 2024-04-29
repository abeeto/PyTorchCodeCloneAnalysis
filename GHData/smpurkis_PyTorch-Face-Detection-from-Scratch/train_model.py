from pathlib import Path

import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import ModelMeta
from models.PoolResnet import PoolResnet
from models.Resnet import Resnet
from models.MobilenetV3Backbone import MobilenetV3Backbone

if __name__ == "__main__":
    torch.random.manual_seed(0)

    num_of_patches = nop = 10
    input_shape = (480, 480)
    filters = 128
    # filters = 576
    lr = 1e-4

    # name = f"pretrained_mobilenetv3backbone_{filters}_{nop}x{nop}_{input_shape[0]}x{input_shape[1]}_sam_adam_all_data2"
    name = f"custom_poolresnet_{filters}_{nop}x{nop}_{input_shape[0]}x{input_shape[1]}_sam_adam_all_data2"
    log_path = Path(f"logs/out_{name}.log")
    log_path.unlink(missing_ok=True)
    model_save_path = f"./saved_models/{name}.pth"

    model = PoolResnet(
        filters=filters,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
        num_of_residual_blocks=10,
    ).cuda()

    model.summary()
    model_setup = ModelMeta(
        model=model,
        lr=lr,
        log_path=log_path,
    )

    # checkpoint = torch.load("lightning_logs/pretrained_mobilenetv3backbone_576_15x15_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt")
    # checkpoint = torch.load("lightning_logs/custom_poolresnet_128_10x10_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt")
    # checkpoint = torch.load("lightning_logs/custom_resnet_64_15x15_480x480_sam_adam/checkpoints/epoch=52-step=42611.ckpt")
    # model_setup.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        gpus=1 if device == torch.device("cuda") else 0,
        max_epochs=70,
        precision=16,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
    )
    dm = WIDERFaceDataModule(
        batch_size=8,
        input_shape=input_shape,
        num_of_patches=num_of_patches,
        shuffle=False,
    )
    trainer.fit(model=model_setup, datamodule=dm)
    model_setup.to_torchscript(model_save_path)
    # trainer.test(model=model_setup, test_dataloaders=dm.test_dataloader())
