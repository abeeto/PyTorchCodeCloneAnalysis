
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ignite.handlers import EarlyStopping
from torchsummary import summary
from tensorboardX import SummaryWriter

import os
import time
import datetime
import shutil
import sys

from opts import args
from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint, plot_figs
from data_loader import get_train_loader, get_test_loader
from models import *
from tiny_imagenet_data_loader import tiny_image_data_loader

# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def validate(val_loader, model, criterion):
    model.eval()
    all_acc = []
    all_acc_top5 = []
    all_loss = []
    
    if args.model == "Elastic_InceptionV3":
        for ix in range((num_outputs-1)):
            all_loss.append(AverageMeter())
            all_acc.append(AverageMeter())
            all_acc_top5.append(AverageMeter())
    else:
        for ix in range(num_outputs):
            all_loss.append(AverageMeter())
            all_acc.append(AverageMeter())      
            all_acc_top5.append(AverageMeter())  
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)     
        
        losses = 0
        
        outputs = model(input_var)
        with torch.no_grad():
            for ix in range(len(outputs)):
                loss = criterion(outputs[ix], target_var)
                all_loss[ix].update(loss.item(), input.size(0))
                losses += loss
                
                prec1 = accuracy(outputs[ix].data, target)
                all_acc[ix].update(prec1[0].data[0].item(), input.size(0))

                # top 5 accuracy
                prec5 = accuracy(outputs[ix].data, target, topk=(5,))
                all_acc_top5[ix].update(prec5[0].data[0].item(), input.size(0))
    accs = []
    ls = []
    accs_top5 = []
    for i, j, k in zip(all_acc, all_loss, all_acc_top5):
        accs.append(float(100-i.avg))
        ls.append(j.avg)
        accs_top5.append(float(100-k.avg))
    print("validation top 5 error: ", accs_top5)
    return accs, ls, accs_top5


def train(train_loader, model, criterion, optimizers, epoch):

    model.train()

    lr = None
    all_acc = []
    all_acc_top5 = []
    all_loss = []

    for ix in range(num_outputs):
        all_loss.append(AverageMeter())
        all_acc.append(AverageMeter())
        all_acc_top5.append(AverageMeter())

    LOG("==> train ", logFile)
    # print("num_outputs: ", num_outputs)
    
    for i, (input, target) in enumerate(train_loader):
        # print("input: ", input, input.shape)
        # print("target: ", target, target.shape)



        # bp_1
        if args.backpropagation == 1:
            # LOG("enter backpropagation method : " + str(args.backpropagation) +"\n", logFile)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            
              
            for ix in range(num_outputs):
                outputs = model(input_var)      
                # 
                optimizers[ix].zero_grad()

                loss = criterion(outputs[ix], target_var)

                loss.backward()
                
                optimizers[ix].step()

                # optimizer.zero_grad()
                # if ix == (num_outputs - 1):
                #     loss.backward()
                # else:
                #     loss.backward(retain_graph=True)

                # optimizer.step()
                all_loss[ix].update(loss.item(), input.size(0))

                # top 1 accuracy
                prec1 = accuracy(outputs[ix].data, target)
                all_acc[ix].update(prec1[0].data[0].item(), input.size(0))

                # # top 5 accuracy
                prec5 = accuracy(outputs[ix].data, target, topk=(5,))
                # print("prec top 5-1: ", prec5)
                # print("prec top 5-2: ", prec5[0])
                # print("prec top 5-3: ", prec5[0].data[0].item())
                all_acc_top5[ix].update(prec5[0].data[0].item(), input.size(0))            

        # elif args.backpropagation == 2:
        #     # LOG("enter backpropagation method : " + str(args.backpropagation) +"\n", logFile)
        #     #　bp_2 
        #     for ix in range(num_outputs):
                
        #         target = target.cuda(async=True)
        #         input_var = torch.autograd.Variable(input)
        #         target_var = torch.autograd.Variable(target)
        #         optimizer.zero_grad()
        #         outputs = model(input_var)
        #         loss = criterion(outputs[ix], target_var)
        #         loss.backward()
        #         optimizer.step()
        
        #         all_loss[ix].update(loss.item(), input.size(0))
        
        #         # top 1 accuracy
        #         prec1 = accuracy(outputs[ix].data, target)
        #         all_acc[ix].update(prec1[0].data[0].item(), input.size(0))
        
        #         # top 5 accuracy
        #         prec5 = accuracy(outputs[ix].data, target, topk=(5,))
        #         all_acc_top5[ix].update(prec5[0].data[0].item(), input.size(0))

        # elif args.backpropagation == 3:
        #     # LOG("enter backpropagation method : " + str(args.backpropagation) +"\n", logFile)
        #     # bp_3
        #     target = target.cuda(async=True)
        #     input_var = torch.autograd.Variable(input)
        #     target_var = torch.autograd.Variable(target)

        #     optimizer.zero_grad()
        #     outputs = model(input_var)
        #     losses = 0
        #     for ix in range(len(outputs)):
        #         # print("outputs[ix]: ", outputs[ix])
        #         loss = criterion(outputs[ix], target_var)
        #         losses += loss

        #         all_loss[ix].update(loss.item(), input.size(0))
            
        #         # top 1 accuracy
        #         prec1 = accuracy(outputs[ix].data, target)
        #         all_acc[ix].update(prec1[0].data[0].item(), input.size(0))

        #         # top 5 accuracy
        #         prec5 = accuracy(outputs[ix].data, target, topk=(5,))
        #         all_acc_top5[ix].update(prec5[0].data[0].item(), input.size(0))
            
        #     # losses = losses/len(outputs)
        #     losses.backward()
        #     optimizer.step()
        else:
            NotImplementedError

        
    accs = []
    accs_top5 = []
    ls = []
    for i, j, k in zip(all_acc, all_loss, all_acc_top5):
        accs.append(float(100-i.avg))
        ls.append(j.avg)
        accs_top5.append(float(100-k.avg))

    try:
        lr = float(str(optimizers[-1]).split("\n")[-5].split(" ")[-1])
    except:
        lr = 100
    
    print("train epoch top 5 error: ", accs_top5)
    return accs, ls, lr, accs_top5


def main(**kwargs):
    global args
    lowest_error1 = 100
    
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    program_start_time = time.time()
    instanceName = "Classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + args.model
    
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model_name + os.sep + ts_str
    
    tensorboard_folder = path + os.sep + "Graph"
    os.makedirs(path)
    args.savedir = path

    writer = SummaryWriter(tensorboard_folder)

    global logFile
    logFile = path + os.sep + "log.txt"
    args.filename = logFile
    global num_outputs
    
    print(args)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == "cifar100" or args.data == "CIFAR100":
        fig_title_str = " on CIFAR-100"

    elif args.data == "cifar10" or args.data == "CIFAR10":
        fig_title_str = " on CIFAR-10"
    elif args.data == "tiny_imagenet":
        fig_title_str = " on tiny_imagenet"
    else:
        LOG("ERROR =============================dataset should be CIFAR10 or CIFAR100", logFile)
        NotImplementedError

    captionStrDict = {
        "fig_title" : fig_title_str,
        "x_label" : "epoch",
        'elastic_final_layer_label': "Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Intermediate_Layer_Classifier_"
    }

    # save input parameters into log file

    LOG("program start time: " + ts_str +"\n", logFile)


    # if args.layers_weight_change == 1:
    #     LOG("weights for intermediate layers: 1/(34-Depth), giving different weights for different intermediate layers output, using the formula weigh = 1/(34-Depth)", logFile)
    # elif args.layers_weight_change == 0:
    #     LOG("weights for intermediate layers: 1, giving same weights for different intermediate layers output as  1", logFile)
    # else:
    #     print("Parameter --layers_weight_change, Error")
    #     sys.exit()   
    
    if args.model == "Elastic_ResNet18" or args.model == "Elastic_ResNet34" or args.model == "Elastic_ResNet50" or args.model == "Elastic_ResNet101" or args.model == "Elastic_ResNet152":
        model = Elastic_ResNet(args, logFile)

    elif args.model == "Elastic_InceptionV3":
        args.target_size = (299, 299, 3) # since pytorch inceptionv3 pretrained accepts image size (299, 299, 3) instead of (224, 224, 3)
        model = Elastic_InceptionV3(args, logFile)

    elif args.model == "Elastic_MobileNet":
        model = Elastic_MobileNet(args, logFile)

    elif args.model == "Elastic_VGG16":
        model = Elastic_VGG16_bn(args, logFile)
    
    elif args.model == "Elastic_SqueezeNet":
        model = Elastic_SqueezeNet(args, logFile)
        
    elif args.model == "Elastic_DenseNet121" or args.model == "Elastic_DenseNet169" or args.model == "Elastic_DenseNet201":
        model = Elastic_DenseNet(args, logFile)

    else:
        LOG("--model parameter should be in ResNet, InceptionV3, MobileNet, VGG16, SqueezeNet, DenseNet", logFile)
        exit()    
    
    num_outputs = model.num_outputs
    # num_outputs = 1

    LOG("num_outputs: " + str(num_outputs), logFile)
    LOG("successfully create model: " + args.model, logFile)

    args_str = str(args)
    LOG(args_str, logFile)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # TUT thinkstation data folder path
    data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200"

    # narvi data folder path
    # data_folder = "/home/zhouy/data/tiny-imagenet-200"

    # XPS 15 laptop data folder path
    # data_folder = "D:\Elastic\data"
    # args.batch_size = 1


    summary(model, (3,224,224))


    if args.data == "tiny_imagenet":
        train_loader, test_loader = tiny_image_data_loader(data_folder, args)
    else:
        train_loader = get_train_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False, target_size = args.target_size,
                                                        random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
                                                        num_workers=4,pin_memory=True, debug=args.debug)


        test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = args.target_size,
                                        num_workers=4,pin_memory=True, debug = args.debug)
    
    criterion = nn.CrossEntropyLoss().cuda()

    if args.data != "tiny_imagenet":
        pretrain_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.pretrain_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        LOG("==> Pretraining for **1** epoches    \n", logFile)
        for pretrain_epoch in range(0, 1):
            accs, losses, lr = train(train_loader, model, criterion, pretrain_optimizer, pretrain_epoch)
            epoch_result = "    pretrain epoch: " + str(pretrain_epoch) + ", pretrain error: " + str(accs) + ", pretrain loss: " + str(losses) + ", pretrain learning rate: " + str(lr) + ", pretrain total train sum loss: " + str(sum(losses))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            LOG(epoch_result, logFile)

        summary(model, (3,224,224))

    LOG("==> Full training    \n", logFile)
    for param in model.parameters():
        param.requires_grad = True
    
    optimizers = []
    childs = []
    k = 0
    for child in model.parameters():
        childs.append(child)
        k += 1
    
    # childs_params = [childs[:9], childs[:15], childs[:21], childs[:27], 
    #                     childs[:33], childs[:39], childs[:45], childs[:51],
    #                     childs[:57], childs[:63], childs[:69], childs[:75], childs]
    childs_params = [childs[:25], childs[:43], childs[:61], childs]

    for i in range(num_outputs):
        optimizer = torch.optim.SGD(childs_params[i], args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizers.append(optimizer)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    # summary(model, (3,224,224))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-4, patience=10)
    
    # implement early stop by own
    EarlyStopping_epoch_count = 0

    epochs_train_accs = []
    epochs_train_top5_accs = []
    epochs_train_losses = []
    epochs_test_accs = []
    epochs_test_losses = []
    epochs_lr = []
    epochs_test_top5_accs = []

    for epoch in range(0, args.epochs):
        
        epoch_str = "==================================== epoch %d ==============================" % epoch
        LOG(epoch_str, logFile)
        # Train for one epoch
        accs, losses, lr, accs_top5 = train(train_loader, model, criterion, optimizers, epoch)
        epochs_train_accs.append(accs)
        epochs_train_losses.append(losses)
        epochs_lr.append(lr)
        epochs_train_top5_accs.append(accs_top5)


        writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'lr', lr, epoch)
        for i, a, l, k in zip(range(len(accs)), accs, losses, accs_top5):
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_error_' + str(i), a, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_losses_' + str(i), l, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_top5_error_' + str(i), k, epoch)
        
        epoch_result = "\ntrain error: " + str(accs) + "top 5 error: " + str(accs_top5) + ", \nloss: " + str(losses) + ", \nlearning rate " + str(lr) + ", \ntotal train sum loss " + str(sum(losses))
        LOG(epoch_result, logFile)

        if num_outputs > 1:
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_total_sum_losses', sum(losses), epoch) 
            losses.append(sum(losses)) # add the total sum loss
            LOG("train_total_sum_losses: " + str(sum(losses)), logFile)              
        
        # run on test dataset
        LOG("==> test \n", logFile)
        test_accs, test_losses, test_top5_accs = validate(test_loader, model, criterion)
        
        epochs_test_accs.append(test_accs)
        epochs_test_losses.append(test_losses)
        epochs_test_top5_accs.append(test_top5_accs)

        for i, a, l, k in zip(range(len(test_accs)), test_accs, test_losses, test_top5_accs):
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_error_' + str(i), a, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_losses_' + str(i), l, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_top5_losses_' + str(i), k, epoch)

        test_result_str = "==> Test epoch: \nfinal output classifier error: " + str(test_accs) + "test top 5 error: " + str(test_top5_accs) + ", \ntest_loss" +str(test_losses) + ", \ntotal test sum loss " + str(sum(test_losses))
        LOG(test_result_str, logFile)
        
        total_loss = sum(test_losses)
        
        if num_outputs > 1:
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_total_sum_losses', total_loss, epoch) 
            test_losses.append(total_loss) # add the total sum loss
            LOG("test_total_sum_losses: " + str(total_loss), logFile)   
        
        log_stats(path, accs, losses, lr, test_accs, test_losses, accs_top5, test_top5_accs)

        # Remember best prec@1 and save checkpoint
        is_best = test_accs[-1] < lowest_error1 #error not accuracy, but i don't want to change variable names
        
        if is_best:
            
            lowest_error1 = test_accs[-1]  #但是有个问题，有时是倒数第二个CLF取得更好的结果
            
            save_checkpoint({
                'epoch': epoch,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_prec1': lowest_error1,
                'optimizer': optimizer.state_dict(),
            }, args)

        # apply early_stop with monitoring val_loss
        # EarlyStopping(patience=15, score_function=score_function(val_loss), trainer=model)
        
        scheduler.step(total_loss) # adjust learning rate with test_loss
        
        if epoch == 0:
            prev_epoch_loss = total_loss # use all intemediate classifiers sum loss instead of only one classifier loss
        else:
            if total_loss >= prev_epoch_loss: # means this current epoch doesn't reduce test losses
                EarlyStopping_epoch_count += 1
        if EarlyStopping_epoch_count > 20:
            LOG("No improving test_loss for more than 10 epochs, stop running model", logFile)
            break

    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # FLOPS_result = 'Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6)
    # LOG(FLOPS_result, logFile)
    # print(FLOPS_result)
    writer.close()

    end_timestamp = datetime.datetime.now()
    end_ts_str = end_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    LOG("program end time: " + end_ts_str +"\n", logFile)

    # here plot figures
    plot_figs(epochs_train_accs, epochs_train_losses, epochs_test_accs, epochs_test_losses, args, captionStrDict)
    LOG("============Finish============", logFile)

if __name__ == "__main__":

    main()
