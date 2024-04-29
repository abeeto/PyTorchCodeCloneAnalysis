import argparse
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.centerface import Centerface
from src.centerface_loss import CenterfaceLoss
from preprocessing.datapreprocessing import TrainAugmentation, EvalAugmentation
from datasets.widerface import WiderDataset

parser = argparse.ArgumentParser(
    description='Centerface Detector Training With Pytorch')

parser.add_argument('--dataset_dir', default='data/', help='Dataset directory path')
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('--p_channels', default=24, type=int,
                    help='Pyramid level channels')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float,
#                     help='Momentum value for optim')
# parser.add_argument('--weight_decay', default=5e-4, type=float,
#                     help='Weight decay for SGD')
parser.add_argument('--pretrained', default=None, type=str, help='Pre-trained base model')
parser.add_argument('--base_pretrained', default=None, type=str, help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--val_epochs', default=10, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=1, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input_size', default=640, type=int, help='Input images size to the net')
parser.add_argument('--down_stride', default=4, type=int, help='Stride of network')
parser.add_argument('--l_offset', default=1., type=float,
                    help='Weight variable for offset loss')
parser.add_argument('--l_scale', default=0.1, type=float,
                    help='Weight variable for scale loss')
parser.add_argument('--l_landmark', default=0.1, type=float,
                    help='Weight variable for landmark loss')

args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if DEVICE == 'cuda'
#     torch.backends.cudnn.benchmark = True

print(args)

net = Centerface(args.p_channels, args.width_mult)
net.to(DEVICE)

if args.pretrained:
    net.load_state_dict(torch.load(args.pretrained))
elif args.base_pretrained:
    net.load_base_state_dict(torch.load(args.base_pretrained))

if args.resume:
    pass

train_transform = TrainAugmentation(args.input_size, args.down_stride)
val_transform = EvalAugmentation(args.input_size)

print("Prepare training dataset.")
train_dataset = WiderDataset(args.dataset_dir, mode='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
             num_workers=args.num_workers, shuffle=True)
print("Prepare validation dataset.")
val_dataset = WiderDataset(args.dataset_dir, mode='val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
             num_workers=args.num_workers, shuffle=False)

# todo, Focal loss(heatmap), Smooth_L1 for (offset, scale, landmarks)
criterion = CenterfaceLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

def outputs_to_dict(heatmap, scales, offsets, landmarks):
    return {'hm':heatmap, 'scale':scales, 'off':offsets}

def train(loader, net, criterion, loss_weights, optimizer, device, debug_steps=100, epoch=0):
    net.train(True)
    running_loss = 0
    running_hm_loss = 0
    running_off_loss = 0
    running_s_loss = 0
    for i, data in tqdm(enumerate(loader)):
        batch = data
        images = batch['input'].to(device)

        optimizer.zero_grad()
        heatmap, scales, offsets, landmarks = net(images)
        output = outputs_to_dict(heatmap, scales, offsets, landmarks)

        hm_loss, s_loss, off_loss = criterion(output, batch)
        loss = loss_weights['scale']*s_loss + loss_weights['off']*off_loss + hm_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_hm_loss = hm_loss.item()
        running_off_loss = off_loss.item()
        running_s_loss = s_loss.item() 
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Hm Loss: {running_hm_loss:.4f}, " +
                f"Scale Loss: {running_s_loss:.4f}, " +
                f"Off Loss: {running_off_loss:.4f}, "
            )
            running_loss = 0.0

def test(loader, net, criterion, loss_weights, device):
    net.eval()
    running_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes = data
        images = images.to(device)
        gt_boxes = boxes.to(device)
        num += 1

        with torch.no_grad():
            heatmap, scales, offsets, landmarks = net(images)
            c_loss, s_loss, off_loss, lms_loss = criterion(gt_boxes, heatmap, scales, offsets, landmarks)
            loss = c_loss + loss_weights['scale']*s_loss + loss_weights['offset']*off_loss + loss_weights['landmark']*lms_loss

        running_loss += loss.item()
    return running_loss / num


loss_weights = {'off':args.l_offset, 'scale':args.l_scale, 'landmark':args.l_landmark}

for epoch in range(args.num_epochs):
    train(train_loader, net, criterion, loss_weights, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
    if epoch % args.val_epochs == 0 or epoch == args.num_epochs - 1:
        val_loss = test(val_loader, net, criterion, loss_weights, DEVICE)
        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")
        model_path = os.path.join(args.checkpoint_folder, f"Epoch-{epoch}-Loss-{val_loss}.pth")
        net.save(model_path)
        print(f"Saved model {model_path}")





