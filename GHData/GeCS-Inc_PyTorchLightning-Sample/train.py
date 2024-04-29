import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything


import model
import data
from utils import EasyDict, plot_dataset

env = EasyDict(
    base_model="efficientnet-b0",
    dataset="custom",  # ["custom"]
    dataset_root="./dataset/",  # "custom" のデータセットの読み込み元
    dataset_download="./external_dataset/",  # image_net 等データセットのダウンロード先
    num_class=3,
    # 学習関連
    batch_size=32,
    learning_rate=1e-3,
    dropout_rate=0.2
)
transform = transforms.Compose([transforms.Resize(400),
                                transforms.RandomCrop(384),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

if __name__ == "__main__":
    seed_everything(42)

    model = model.FineTuningModel(env)

    train_dataset, val_dataset = data.get_dataset(env, transform)

    plot_dataset(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=env.batch_size, num_workers=2)
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=env.batch_size, num_workers=2)

    trainer = Trainer(gpus=1)
    trainer.fit(model, train_dataloader, val_dataloader)
