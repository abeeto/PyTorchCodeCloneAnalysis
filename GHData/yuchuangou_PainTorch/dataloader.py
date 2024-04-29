from dataset import LineColorDataset
from torch.utils.data import DataLoader


def get_train_loader(a_root, b_root, batch_size=64, resize=False, size=(256, 256)):
    data_set = LineColorDataset(a_root, b_root, resize=resize, size=size)
    data_loader = DataLoader(
        dataset=data_set,
        shuffle=True,
        batch_size=batch_size
    )
    return data_loader
