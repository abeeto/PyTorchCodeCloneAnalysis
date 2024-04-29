
from layers import *




class ImageClassificationBase(nn.Module):
    '''
    since we are on the image classification task we use this general class for training and validation steps
    '''
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


###################################################################################################
###################################################################################################

                                 #LeNet for MNIST dataset

###################################################################################################
###################################################################################################


class LeNet(ImageClassificationBase):
    def __init__(self,config):
        super(LeNet, self).__init__()
        self.conv1 = Conv2dBit(1, 6, kernel_size=5, stride=1, padding=0, config=config)
        self.conv2 = Conv2dBit(6, 16, kernel_size=5, stride=1, padding=0, config=config)
        self.fc1   = LinearBit(16*4*4, 120,config=config)
        self.fc2   = LinearBit(120, 84,config=config)
        self.fc3   = LinearBit(84, 10,config=config)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


###################################################################################################
###################################################################################################

                                    # ResNet for CIFAR dataset

###################################################################################################
###################################################################################################

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, config, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dBit(in_planes, planes,kernel_size=3,padding=1,stride=stride,config=config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dBit(planes, planes,kernel_size=3,padding=1,stride=1,config=config)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2dBit(in_planes, self.expansion*planes,kernel_size=1, stride=stride, padding=0,config=config),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ImageClassificationBase):
    def __init__(self, block, num_blocks, config, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dBit(3, 16, kernel_size=3,padding=1,stride=1,config=config)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], config, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], config, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], config, stride=2)
        self.linear = LinearBit(64, num_classes,config=config)


    def _make_layer(self, block, planes, num_blocks, config, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, config, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(config):
    return ResNet(BasicBlock, [3, 3, 3],config)


###################################################################################################
###################################################################################################

                                    # Conv6 for CIFAR dataset

###################################################################################################
###################################################################################################


class Conv6(ImageClassificationBase):

    def __init__(self,config):
        super(Conv6, self).__init__()

        self.layer1 = torch.nn.Sequential(
            Conv2dBit(3, 64, kernel_size=3, stride=1, padding=1,config=config),
            torch.nn.ReLU(),
            Conv2dBit(64, 64, kernel_size=3, stride=1, padding=1,config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            Conv2dBit(64, 128, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            Conv2dBit(128, 128, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            Conv2dBit(128, 256, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            Conv2dBit(256, 256, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = LinearBit(5 * 5 * 256, 256, config)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU())
        self.fc2 = LinearBit(256, 256, config)
        self.layer5 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU())
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc3 = LinearBit(256, 10, config)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc3(out)
        return out


###################################################################################################
###################################################################################################

                                    # VGG for CIFAR dataset

###################################################################################################
###################################################################################################



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(ImageClassificationBase):
    def __init__(self, vgg_name,config):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],config)
        self.classifier = LinearBit(512, 10,config=config)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg,config):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2dBit(in_channels, x, kernel_size=3, padding=1,stride=1,config=config),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




###################################################################################################
###################################################################################################

                                    # EfficientNet for CIFAR dataset

###################################################################################################
###################################################################################################



def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels,config):
        super(SE, self).__init__()
        self.se1 = Conv2dBit(in_channels, se_channels,
                             kernel_size=1,
                               stride=1,
                               padding=0, config=config)
        self.se2 = Conv2dBit(se_channels, in_channels,
                             kernel_size=1,
                               stride=1,
                               padding=0, config=config)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 config,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = Conv2dBit(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0, config=config)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = Conv2dBit(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               config=config)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels,config)

        # Output
        self.conv3 = Conv2dBit(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0, config=config)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(ImageClassificationBase):
    def __init__(self, cfg, config,num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = Conv2dBit(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1, config=config)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32,config=config)
        self.linear = LinearBit(cfg['out_channels'][-1], num_classes, config=config)

    def _make_layers(self, in_channels,config):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          config,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def EfficientNetB0(config):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg,config)
