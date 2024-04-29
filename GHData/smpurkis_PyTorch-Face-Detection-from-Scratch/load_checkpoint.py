import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import BaseModel, ModelMeta

if __name__ == "__main__":
    checkpoint = torch.load(
        "lightning_logs/version_208/checkpoints/epoch=50-step=10250.ckpt"
    )

    num_of_patches = 5
    input_shape = (320, 320)
    model = BaseModel(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
        num_of_residual_blocks=10,
    ).cuda()
    model.summary()
    model_setup = ModelMeta(model=model, lr=1e-4)
    model_setup.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = WIDERFaceDataModule(
        batch_size=8,
        input_shape=input_shape,
        num_of_patches=num_of_patches,
        shuffle=False,
    )
    dm.setup()
    tensor, target = dm.val_dataset[0]
    tensor = tensor.reshape(-1, *dm.val_dataset[0][0].shape).cuda()
    target = target.cuda()
    gt_bbx = model_setup.model.reduce_bounding_boxes(target)
    out = model_setup(tensor)
    pred_bbx = model_setup.model.reduce_bounding_boxes(out[0])
    print(dm.val_dataset[0])
    # trainer.fit(model=model_setup, datamodule=dm)
