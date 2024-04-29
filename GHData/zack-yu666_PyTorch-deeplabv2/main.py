import os

import itertools

import torch

import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import transforms as ext_transforms
from test import Test
from metric.iou import IoU
from args import get_arguments
import utils
from deeplab_multi import Encoder,Decoder
from loss import CrossEntropy2d

# Get the arguments
args = get_arguments()

# Set the cuda mode
device = torch.device(args.device)            
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

# Learning rate adjustment function for the optimizers
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.max_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.max_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10




def main():    
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        mode='train',
        max_iters=args.max_iters,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    
    trainloader_iter = enumerate(train_loader)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        max_iters=args.max_iters,
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        max_iters=args.max_iters,
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding
    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get the parameters for the validation set 
    if args.mode.lower() == 'test':
        images,labels = iter(test_loader).next()
    else:
        images,labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
   
    
    print("\nTraining...\n")

    num_classes = len(class_encoding)
    # Define the model with the encoder and decoder from the deeplabv2
    input_encoder = Encoder().to(device)
    decoder_t=Decoder( num_classes).to(device)

    # Define the entropy loss for the segmentation task
    criterion = CrossEntropy2d()
    

    # Set the optimizer function for model
    optimizer_g = optim.SGD(itertools.chain(input_encoder.parameters(),decoder_t.parameters()), lr=args.learning_rate, momentum=0.9,weight_decay=1e-4)

    optimizer_g.zero_grad()

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        
        input_encoder,decoder_t, optimizer_g, start_epoch, best_miou = utils.load_checkpoint(
          input_encoder,decoder_t,optimizer_g, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
        
    metric.reset()
    
    val = Test(input_encoder,decoder_t, val_loader, criterion, metric, device)
    
    for i_iter in range(args.max_iters):

        
   
        optimizer_g.zero_grad()
        adjust_learning_rate(optimizer_g, i_iter)
        
        _, batch_data = trainloader_iter.__next__()
        inputs = batch_data[0].to(device)
        labels = batch_data[1].to(device)
        
        
        f_i=input_encoder(inputs)

        outputs_i=decoder_t(f_i)
        loss_seg=criterion(outputs_i, labels)
    
        loss_g = loss_seg
        loss_g.backward()
        optimizer_g.step()
        
        
        
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(
            i_iter, args.max_iters, loss_g))
            print(">>>> [iter: {0:d}] Validation".format(i_iter))

            # Validate the trained model after the weights are saved
            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [iter: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(i_iter, loss, miou))
            
            if miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(input_encoder,decoder_t,optimizer_g, i_iter + 1, best_miou,
                                      args)
            
        
if __name__ == '__main__':
    main()
