from torch.utils.data import DataLoader

from dataprocess.StyleTransferDataset import StyleTransferDataset
from resources.utilities import *
from Model.AdaIN import AdaIN
import pprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FOR TEST PURPOSES
if __name__ == "__main__":

    style_layers_req = ["ReLu_1", "ReLu_2", "ReLu_3", "ReLu_4"]

    # TRAIN
    transformed_dataset = StyleTransferDataset(CONTENT_PATH, STYLE_PATH, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=1,
                            shuffle=True)

    sample = next(iter(dataloader))

    adain = AdaIN(4, style_layers_req)
    adain.load_save()

    result, _ = adain.forward(sample['style'].to(device), sample['content'].to(device))
    show_tensor(sample['content'], "content")
    show_tensor(sample['style'], "style")
    show_tensor(result, "res")
