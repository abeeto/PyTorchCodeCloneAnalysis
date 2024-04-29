from torch.utils.tensorboard import SummaryWriter
from data import get_data
import torchvision
import torch


dl_train, dl_test = get_data()
dl_train_ = iter(dl_train)
imgs, lbls = dl_train_.next()
writer = SummaryWriter('./tensorboard')
writer.add_image('images', torchvision.utils.make_grid(imgs))

# writer.add_graph(model, imgs)

# lbls_ = [dl_train.dataset.classes[l] for l in lbls]
# _, embeds = model(imgs.float())
# writer.add_embedding(embeds, metadata=lbls_)
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# select random images and their target indices
images, labels = select_n_random(dl_train.dataset.data, dl_train.dataset.targets)
classes = dl_train.dataset.classes
# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.close()
