from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import backbone_utils
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import boxes
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')

    parser.add_argument('--data_path', default= r'F:\DataSource\COCO2014\train2014', help='dataset path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--b', '--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--output-dir', default='./result', help='path where to save')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--distributed', default=True, help='if distribute or not')
    parser.add_argument('--parallel', default=False, help='if distribute or not')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args

def vgg16_fpn_backbone(
    backbone_name,
    pretrained,
    fpn,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    not_trainable_layers=10,
    returned_layers=None,
    extra_blocks=None
):
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Args:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default a ``LastLevelMaxPool`` is used.
    """
    backbone = models.vgg16(pretrained=pretrained).features
    # select layers that wont be frozen
    assert 0 <= not_trainable_layers <= 30

    # 固定VGG16的前10层，后面网络的参数才训练
    for layer in backbone[:not_trainable_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            # 根据不同的VGG，设置不同的returned_layers, VGG16中features特征层一共30层
            # FPN特征金字塔层，选择VGG中不同层级的特征尺寸的输出的索引，以下几个索引对应的卷积层输出的特征分别是 64，128，256， 512
            returned_layers = [2, 7, 14, 28]
        assert min(returned_layers) > 0 and max(returned_layers) < 31
        return_layers = {f'{k}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[i].out_channels for i in returned_layers]

        return backbone_utils.BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels,
                                              extra_blocks=extra_blocks)
    else:
        m = nn.Sequential(
            backbone,
            # depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m



model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    ,'fasterrcnn_vgg16_fpn_coco':
        ''
}
def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
            module.eps = eps

def fasterrcnn_vgg16_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    not_trainable_backbone_layers = backbone_utils._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 30, 10)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = vgg16_fpn_backbone('vgg16', pretrained_backbone, fpn=True, not_trainable_layers=not_trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


def collate_fn_coco(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    args = get_args()

    sys_type = 'win'
    num_classes = 91
    gpu_id = 0
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    path = {
        "linux":[r'D:\DataSource\COCO2014\train2014', r'D:\DataSource\COCO2014\annotations_trainval2014\annotations\instances_train2014.json'],
        "win":[r'D:\DataSource\COCO2014\train2014', r'D:\DataSource\COCO2014\annotations_trainval2014\annotations\instances_train2014.json']
    }
    data_dir, anno_path = path[sys_type][0], path[sys_type][1]
    models.detection.fasterrcnn_resnet50_fpn()
    model = fasterrcnn_vgg16_fpn()
    # model.eval()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # test code
    # images = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # output = model(images)
    # print(output)

    coco_dataset = datasets.CocoDetection(root=data_dir, annFile=anno_path, transform=transforms.ToTensor())
    sampler = RandomSampler(coco_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=1, drop_last=True)
    # batch_size = 1
    data_loader = DataLoader(coco_dataset, batch_sampler=batch_sampler, num_workers=1, collate_fn=collate_fn_coco)

    len_dataloader = len(data_loader)

    pbar = tqdm(data_loader)
    for epoch in range(args.epochs):
        model.train()
        i = 0

        for imgs, annotations in pbar:
            total_loss = 0
            rpn_loc_loss = 0
            rpn_cls_loss = 0
            roi_loc_loss = 0
            roi_cls_loss = 0
            iteration = 0
            imgs = list(img.to(device) for img in imgs)
            targets = []
            for anno in annotations:
                d = {}
                instance_bbox = []
                instance_cls = []
                for bb in anno:
                    xyxy = boxes.box_convert(torch.tensor(bb['bbox']), 'xywh', 'xyxy')
                    class_id = bb['category_id']
                    instance_bbox.append(xyxy)
                    instance_cls.append(class_id)
                # d['boxes'] = torch.tensor([boxes.box_convert(torch.tensor(an['bbox']), 'xywh', 'xyxy') for an in
                # anno]).to(device) d['labels'] = torch.tensor([an['category_id'] for an in anno]).to(device)
                d['boxes'] = torch.stack(instance_bbox, dim=0).to(device)
                d['labels'] = torch.tensor(instance_cls).to(device)
                targets.append(d)
            print(targets)

            # i += 1
            # imgs = list(img.to(device) for img in imgs)
            # annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, targets)
            roi_cls = loss_dict['loss_classifier']
            roi_loc = loss_dict['loss_box_reg']
            rpn_cls = loss_dict['loss_objectness']
            rpn_loc = loss_dict['loss_rpn_box_reg']

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'rpn_loc': rpn_loc_loss / (iteration + 1),
                                'rpn_cls': rpn_cls_loss / (iteration + 1),
                                'roi_loc': roi_loc_loss / (iteration + 1),
                                'roi_cls': roi_cls_loss / (iteration + 1),
                                # 'lr': get_lr(optimizer)
                                })

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')