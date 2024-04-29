# single fn train
# single fn val
# loader sampler distributed
# save if master process
# label non_blocking = True
# build_model
# export CUDA_VISIBLE_DEVICES
# pyvls utils distributed.py line 33 -> port

from comet_ml import Experiment
import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import matplotlib.pyplot as plt
# PyTorch includes
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
# import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
# import networks.deeplab as deeplab
from encoding.models.danet import DANet
from layers.loss_weighted import SegmentationMultiLosses
from encoding.utils import LR_Scheduler
from dataloaders.helpers import *
from dataloaders.implementation import *
from tqdm import tqdm
from encoding.parallel import DataParallelModel, DataParallelCriterion


exp = Experiment(api_key = 'ijlRL2gSumf01e8d1fkRxaG4O', project_name = 'Attention', workspace = 'ahmedshahin9')


# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
use_sbd = False
nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

task = 'Liver'

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 16  # Training batch size
testBatch = 1  # Testing batch size
useTest = 1  # See evolution of the test set when training?
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 100  # Store a model every snapshot epochs
# relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
nInputChannels = 4  # Number of input channels (RGB + heatmap of extreme points)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 5e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
# net = deeplab.DeepLab(backbone = 'resnet', output_stride = 16, num_classes = 1, sync_bn = False, freeze_bn = True, gate = 4)
net = DANet(1, 'resnet101')
# for name, param in net.named_parameters():
#     if 'backbone' in name:
#         param.requires_grad = False
# train_params = [{'params': net.get_1x_lr_params(), 'lr': p['lr']},
#                 {'params': net.get_10x_lr_params(), 'lr': p['lr'] * 10}]
net = torch.nn.DataParallel(net, device_ids = [0,1,2,3])
# if resume_epoch == 0:
#     # print("Initializing from pretrained Deeplab-v2 model")
# else:
#     print("Initializing weights from: {}".format(
#         os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
#     # net.load_state_dict(
#     #     torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
#     #                map_location=lambda storage, loc: storage))
#     # net.load_state_dict(torch.load('dextr_best.pth', map_location=lambda storage, loc: storage))
#     # net.load_state_dict(torch.load('gate3.pth'))
net.load_state_dict(torch.load('danet_1e-7_91.3.pth'))
# net.load_state_dict(torch.load('DANet101.pth.tar')['state_dict'], strict = False)
print("No of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))


net.to(device)

# Training the network
if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)


    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    criterion = SegmentationMultiLosses(1)
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.CropFromMaskStatic(crop_elems=('image', 'gt'), relax = 50, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt', is_val = False),
        # tr.ToImage(norm_elem='extreme_points'),
        # tr.NEllipse(is_val = False),
        tr.NEllipseWithGaussians(alpha = 0.6, is_val = False),
        # tr.AddConfidenceMap(elem = ('crop_image'), hm_type = 'l1l2', tau = 7),
        tr.ConcatInputs(elems=('crop_image', 'nellipseWithGaussians')),
        tr.ToTensor()])
    composed_transforms_ts = transforms.Compose([
    #     tr.CreateBBMask(),
        tr.CropFromMaskStatic(crop_elems=('image', 'gt'), relax = 50, zero_pad=zero_pad_crop),
        tr.FixedResize(resolutions={'void_pixels': None, 'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt', is_val = True),
        # tr.ToImage(norm_elem='extreme_points'),
        # tr.NEllipse(is_val = True),
        tr.NEllipseWithGaussians(alpha = 0.6, is_val = True),
        # tr.AddConfidenceMap(elem = ('crop_image'), hm_type = 'l1l2', tau = 7),
        tr.ConcatInputs(elems=('crop_image', 'nellipseWithGaussians')),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split = 'train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split = 'val', transform=composed_transforms_ts)

    if use_sbd:
        sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
        db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
    else:
        db_train = voc_train

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
    p['dataset_test'] = str(db_train)
    p['transformations_test'] = [str(tran) for tran in composed_transforms_ts.transforms]

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2, drop_last = True)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=2)

    # scheduler = LR_Scheduler('poly', p['lr'], nEpochs, len(trainloader))

    # trainloader = tqdm(trainloader)
    # testloader = tqdm(testloader)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    # Train variables
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    # aveGrad = 0
    best_val = 0.913
    print("Training Network")
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        print("Current LR", optimizer.param_groups[0]['lr'])

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['concat'], sample_batched['crop_gt']
            assert inputs[:, :3].max() <= 255 and inputs[:, :3].min() >= 0 and len(np.unique(inputs[:, :3].numpy())) > 2
            assert inputs[:,  3].max() <= 255 and inputs[:,  3].min() >= 0 and len(np.unique(inputs[:,  3].numpy())) > 2
            assert gts.min() == 0 and gts.max() == 1 and len(np.unique(gts.numpy())) == 2
            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            output = net.forward(inputs)
            # output = upsample(output, size=(512, 512), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = criterion.forward(output, gts)
            # l1 = class_balanced_cross_entropy_loss(output, gts, size_average=False, batch_average=True)
            # l2 = class_balanced_cross_entropy_loss(att_map2, gts, size_average=False, batch_average=True)
            # l3 = class_balanced_cross_entropy_loss(att_map3, gts, size_average=False, batch_average=True)
            # loss = l1 + 0.1 * l2 + 0.05 * l3
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                tr_loss_for_vis = running_loss_tr
                running_loss_tr = 0

            # Backward the averaged gradient
            # loss /= p['nAveGrad']
            loss.backward()
            # aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            # if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            # scheduler(optimizer, ii, epoch, best_val)
            optimizer.zero_grad()
            aveGrad = 0
        # optimizer.param_groups[0]['lr'] = (epoch * -4.95e-08) + 5e-6
        # optimizer.param_groups[1]['lr'] = (epoch * -4.95e-07) + 5e-5
        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            jac_val = np.zeros(3)
            with torch.no_grad():
                for ii, sample_batched in enumerate(testloader):
                    inputs, gts, gts_full = sample_batched['concat'], sample_batched['crop_gt'], sample_batched['gt']
                    assert inputs[:, :3].max() <= 255 and inputs[:, :3].min() >= 0 and len(np.unique(inputs[:, :3].numpy())) > 2
                    assert inputs[:,  3].max() <= 255 and inputs[:,  3].min() >= 0 and len(np.unique(inputs[:,  3].numpy())) > 2
                    assert gts.min() == 0 and gts.max() == 1 and len(np.unique(gts.numpy())) == 2
                    # Forward pass of the mini-batch
                    inputs, gts = inputs.to(device), gts.to(device)

                    output = net.forward(inputs)
                    # output = upsample(output, size=(512, 512), mode='bilinear', align_corners=True)

                    # Compute the losses, side outputs and fuse
                    loss = criterion.forward(output, gts)
                    # l1 = class_balanced_cross_entropy_loss(output, gts, size_average=False)
                    # l2 = class_balanced_cross_entropy_loss(att_map2, gts, size_average=False)
                    # l3 = class_balanced_cross_entropy_loss(att_map3, gts, size_average=False)
                    # loss = l1 + 0.1 * l2 + 0.05 * l3
                    running_loss_ts += loss.item()

                    # output = output.to(torch.device('cpu'))
                    if ii == 0:
                        att_pos = output[1][0,0].to(torch.device('cpu'))
                        att_ch = output[2][0,0].to(torch.device('cpu'))
                        out  = output[0][0,0].to(torch.device('cpu'))
                        img  = np.transpose(inputs.to(torch.device('cpu'))[0,:3].numpy(), (1,2,0))
                        out  = 1 / (1 + np.exp(-out))
                        plt.subplot(141)
                        plt.imshow(img.astype(int))
                        plt.imshow(colorMaskWithAlpha(gts.to(torch.device('cpu'))[0,0].numpy().astype(float), color='r', transparency=0.3))
                        plt.subplot(142)
                        plt.imshow(out)
                        plt.title("prediction")
                        plt.subplot(143)
                        plt.imshow(att_pos)
                        plt.title("Position Att")
                        plt.subplot(144)
                        plt.imshow(att_ch)
                        plt.title("Channet Att")
                        plt.suptitle("Epoch {0}".format(epoch))
                        exp.log_figure()
                        plt.close()
                        plt.clf()

                    relaxes = [50]
                    thresh = [0.3, 0.5, 0.8]
                    for jj in range(int(inputs.size()[0])):
                        pred = np.transpose(output[0].to(torch.device('cpu')).data.numpy()[jj, :, :, :], (1, 2, 0))
                        pred = 1 / (1 + np.exp(-pred))
                        pred = np.squeeze(pred)
                        gt = tens2image(gts_full[jj, :, :, :])
                        bbox = get_bbox(gt, pad=relaxes[jj], zero_pad=zero_pad_crop)
                        void_pixels = np.squeeze(tens2image(sample_batched['void_pixels']))
                        for k, th in enumerate(thresh):
                            result = (crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop, relax=relaxes[jj]) > th).astype('int')
                            jac_val[k] += calc_jaccard(result, gt, void_pixels)

                    # Print stuff
                    if ii % num_img_ts == num_img_ts - 1:
                        running_loss_ts = running_loss_ts / num_img_ts
                        print('[Epoch: %d, numImages: %5d]' % (epoch, ii*testBatch+inputs.data.shape[0]))
                        print('Loss: %f' % running_loss_ts)
                        jac_avg = jac_val / len(testloader)
                        # writer.add_scalars('data/loss_epoch', {'validation': running_loss_ts, 'training': tr_loss_for_vis}, epoch)
                        # writer.add_scalar('data/validation_accuracy', jac_avg[1], epoch)
                        if jac_avg.max() > best_val:
                            best_val = jac_avg.max()
                            print("SAVING ================================")
                            torch.save(net.state_dict(), 'danet_ft91.3_wtd_loss_nellipse_withGaussian.pth')
                        running_loss_ts = 0
                        print("JACCARD", jac_avg)
                        stop_time = timeit.default_timer()
                        print("Execution time: " + str(stop_time - start_time)+"\n")
    writer.close()
    
def train_epoch(train_loader, cur_epoch):
    start_time = timeit.default_timer()
    print("Current LR", optimizer.param_groups[0]['lr'])
    net.train()
    for ii, sample_batched in enumerate(trainloader):
        inputs, gts = sample_batched['concat'], sample_batched['crop_gt']
        assert inputs[:, :3].max() <= 255 and inputs[:, :3].min() >= 0 and len(np.unique(inputs[:, :3].numpy())) > 2
        assert inputs[:,  3].max() <= 255 and inputs[:,  3].min() >= 0 and len(np.unique(inputs[:,  3].numpy())) > 2
        assert gts.min() == 0 and gts.max() == 1 and len(np.unique(gts.numpy())) == 2
        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        inputs, gts = inputs.to(device), gts.to(device)

            output = net.forward(inputs)
            # output = upsample(output, size=(512, 512), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = criterion.forward(output, gts)
            # l1 = class_balanced_cross_entropy_loss(output, gts, size_average=False, batch_average=True)
            # l2 = class_balanced_cross_entropy_loss(att_map2, gts, size_average=False, batch_average=True)
            # l3 = class_balanced_cross_entropy_loss(att_map3, gts, size_average=False, batch_average=True)
            # loss = l1 + 0.1 * l2 + 0.05 * l3
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                tr_loss_for_vis = running_loss_tr
                running_loss_tr = 0

            # Backward the averaged gradient
            # loss /= p['nAveGrad']
            loss.backward()
            # aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            # if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            # scheduler(optimizer, ii, epoch, best_val)
            optimizer.zero_grad()
            aveGrad = 0
    
