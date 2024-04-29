import os
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from DataSet import MyDataset
from Metrics import Iou
from Transforms import *
from models.Enet.Enet import ENet
from Criterion import WeightCrossEntropyLoss
from args import MyArgumentParser
from Utils import save_checkpoint, load_checkpoint, data_classes

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 设定使用第2张显卡

args = MyArgumentParser()
device = torch.device(args.device)


def load_dataset():
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)

    image_transform = Compose(
        [Resize(args.height, args.width),
         Normalize(mean=args.mean, std=args.std),
         ToTensor()])

    train_set = MyDataset(
        args.dataset_dir,
        transform=image_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = MyDataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # # Load the test set as tensors
    # test_set = dataset(
    #     args.dataset_dir,
    #     mode='test',
    #     transform=image_transform,
    #     label_transform=label_transform)
    # test_loader = data.DataLoader(
    #     test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers)

    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    if args.weights is not None:
        class_weights = torch.from_numpy(np.array(args.weights)).float().to(device)
        # Set the weight of the unlabeled class to 0
        # if args.ignore_unlabeled:
        #     ignore_index = list(class_encoding).index('unlabeled')
        #     class_weights[ignore_index] = 0
        print("Class weights:", class_weights)
    else:
        class_weights = None
    return (train_loader, val_loader), class_weights


def train(train_loader, val_loader, class_weights, data_classes):
    print("\nTraining...\n")

    model = ENet(args.num_classes).to(device)
    print(model)

    criterion = WeightCrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    # if args.ignore_unlabeled:
    #     ignore_index = list(class_encoding).index('unlabeled')
    # else:
    #     ignore_index = None
    metric = Iou(args.num_classes)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    # val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()

        model.train()
        epoch_loss = 0.0
        total_miou = 0.0

        for step, (inputs, labels) in enumerate(train_loader):

            # # Get the inputs and labels
            # data = inputs.numpy()[0].transpose(1,2,0)[:,:,0]
            # pd.DataFrame(data).to_csv('./1.csv')
            # plt.imshow(inputs.numpy()[0].transpose(1,2,0))
            # plt.show()
            # exit(0)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # Forward propagation
            outputs = model(inputs)

            # Loss computation
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            iou, miou = metric.IOU(labels.detach(), outputs.detach())
            total_miou += miou

            if args.print_step and (step + 1) % args.print_step == 0:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        epoch_loss = epoch_loss / len(train_loader)
        total_miou /= len(train_loader)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, total_miou))
        # Print per class IoU on last epoch or if best iou
        if total_miou > best_miou:
            for key, class_iou in zip(data_classes, iou):
                print("{0}: {1:.4f}".format(key, class_iou))

        # Save the model if it's the best thus far
        if total_miou > best_miou:
            print("\nBest model thus far. Saving...\n")
            best_miou = total_miou
            save_checkpoint(model, optimizer, epoch + 1, best_miou,
                            args)

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            # loss, (iou, miou) = val.run_epoch(args.print_step)
            model.eval()
            val_epoch_loss = 0.0
            val_total_miou = 0.0
            for step, (inputs, labels) in enumerate(val_loader):
                # Get the inputs and labels
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                with torch.no_grad():
                    # Forward propagation
                    outputs = model(inputs)
                    # Loss computation
                    loss = criterion(outputs, labels)
                # Keep track of loss for current epoch
                val_epoch_loss += loss.item()
                iou, miou = metric.IOU(labels.detach(), outputs.detach())
                val_total_miou += miou
                # Keep track of evaluation the metric
                # self.metric.add(outputs.detach(), labels.detach())

                if args.print_step and (step+1)%args.print_step == 0:
                    print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

            val_epoch_loss /= len(val_loader)
            val_total_miou /= len(val_loader)
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, val_epoch_loss, val_total_miou))

            # # Print per class IoU on last epoch or if best iou
            # if epoch + 1 == args.epochs or val_total_miou > best_miou:
            #     for key, class_iou in zip(data_classes, iou):
            #         print("{0}: {1:.4f}".format(key, class_iou))
            #
            # # Save the model if it's the best thus far
            # if val_total_miou > best_miou:
            #     print("\nBest model thus far. Saving...\n")
            #     best_miou = val_total_miou
            #     save_checkpoint(model, optimizer, epoch + 1, best_miou,
            #                           args)

    return model





# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    loaders, w_class = load_dataset()

    data_classes, data_cmap = data_classes(args.dataset)

    train_loader, val_loader = loaders
    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, data_classes=data_classes)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = args.num_classes
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        # test(model, test_loader, w_class, class_encoding)