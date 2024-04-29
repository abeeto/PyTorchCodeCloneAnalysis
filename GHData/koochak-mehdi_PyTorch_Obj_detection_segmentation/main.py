import torch as t
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


from utils import utils
from utils.engine import train_one_epoch, evaluate
from dataset import PennFudanDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hiddn_layer = 50
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hiddn_layer,
                                                    num_classes)

    return model

def main(filePath):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    num_classes = 2
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

    indices = t.randperm(len(dataset)).tolist()
    dataset = Subset(dataset, indices[:-50])
    dataset_test = Subset(dataset_test, indices[-50:])

    data_loader = DataLoader(dataset,
                                batch_size=2,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(dataset_test, 
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = t.optim.SGD(params, 
                                lr=.005,
                                momentum=.9,
                                weight_decay=.0005)
    
    lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=.1)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, 10)

        lr_scheduler.step()
        evaluate(model, data_loader_test, device)

    print('---> training finished!')

    t.save(model.state_dict(), filePath)
    print('---> model saved!')

if __name__ == '__main__':
    FILE = 'trainedModel.pth'
    main(FILE)