'''
Training script for VGG-16.
'''
import os
from tqdm import tqdm
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
from image_transforms import RandomRotate, RandomHorizontalFlip, RandomNoise, RandomVerticalFlip, RandomShear, ToTensor
from load_data import LoadDataset
from utils import calculate_accuracy, plot_graph
import train_cfg as t_cfg

def main():
    '''
    Train function.
    '''
    os.environ['TORCH_HOME'] = t_cfg.MODEL_PATH #set the env variable so the model is downloaded inside this folder.

    ########################################################## Model Initialization & Loading ##########################################################
    model_instance = model.Model(model_download_path=t_cfg.MODEL_PATH, new_model_name=t_cfg.MODEL_NAME, input_feature_size=t_cfg.FEATURE_INPUT_SIZE,
                            num_class=t_cfg.NUM_CLASSES)

    vgg_model = model_instance()

    optimizer = Adam(vgg_model.parameters(), lr=t_cfg.LEARNING_RATE) #optimizer
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=t_cfg.LR_DECAY_RATE) #scheduler is used to lower the learning rate during training later.
    loss_criterion = torch.nn.CrossEntropyLoss() #loss function.

    vgg_model = vgg_model.to(t_cfg.DEVICE) #move the network to GPU if available.

    print("--- Model Architecture ---")
    print(vgg_model)




    ########################################################## Data Initialization & Loading ##########################################################
    #Initialize the training data class.
    training_data = LoadDataset(resized_image_size=t_cfg.RESIZED_IMAGE_SIZE, total_images=t_cfg.TOTAL_DATA, classes=t_cfg.CLASSES,
                                data_list=t_cfg.IMG_LABEL_LIST, transform=transforms.Compose([RandomRotate(angle_range=t_cfg.ROTATION_RANGE, prob=t_cfg.ROTATION_PROB),
                                                                                            RandomShear(shear_range=t_cfg.SHEAR_RANGE, prob=t_cfg.SHEAR_PROB),
                                                                                            RandomHorizontalFlip(prob=t_cfg.HFLIP_PROB),
                                                                                            RandomVerticalFlip(prob=t_cfg.VFLIP_PROB),
                                                                                            RandomNoise(mode=t_cfg.NOISE_MODE, prob=t_cfg.NOISE_PROB),
                                                                                            ToTensor(mode='training')]))

    dataloader = DataLoader(training_data, batch_size=t_cfg.BATCH_SIZE, shuffle=t_cfg.DATA_SHUFFLE, num_workers=t_cfg.NUM_WORKERS)




    ########################################################## Model Training & Saving ##########################################################
    best_accuracy = 0

    entire_loss_list = []
    entire_accuracy_list = []

    for epoch_idx in range(t_cfg.EPOCH):

        print("Training for epoch %d has started!"%(epoch_idx+1))

        epoch_training_loss = []
        epoch_accuracy = []
        i = 0
        for i, sample in tqdm(enumerate(dataloader)):

            batch_x, batch_y = sample['image'].to(t_cfg.DEVICE), sample['label'].to(t_cfg.DEVICE)

            optimizer.zero_grad() #clear the gradients in the optimizer between every batch.

            net_output = vgg_model(batch_x) #output from the network.

            total_loss = loss_criterion(input=net_output, target=batch_y)

            epoch_training_loss.append(total_loss.item()) #append the loss of every batch.

            total_loss.backward() #calculate the gradients.
            optimizer.step()

            batch_acc = calculate_accuracy(network_output=net_output, target=batch_y)
            epoch_accuracy.append(batch_acc.cpu().numpy())

        lr_decay.step() #decay rate update
        curr_accuracy = sum(epoch_accuracy)/i
        curr_loss = sum(epoch_training_loss)

        print("The accuracy at epoch %d is %g"%(epoch_idx, curr_accuracy))
        print("The loss at epoch %d is %g"%(epoch_idx, curr_loss))

        entire_accuracy_list.append(curr_accuracy)
        entire_loss_list.append(curr_loss)


        if curr_accuracy > best_accuracy:

            torch.save(vgg_model.state_dict(), t_cfg.SAVE_PATH)
            best_accuracy = curr_accuracy
            print("Model is saved !")


    ########################################################## Graphs ##########################################################
    if t_cfg.PLOT_GRAPH:
        plot_graph(t_cfg.EPOCH, "Epoch", "Training Loss", "Training Loss for %d epoch"%(t_cfg.EPOCH), "./loss.png", [entire_loss_list, 'r--', "Loss"])
        plot_graph(t_cfg.EPOCH, "Epoch", "Training Accuracy", "Training Accuracy for %d epoch"%(t_cfg.EPOCH), "./accuracy.png", [entire_accuracy_list, 'b--', "Accuracy"])

if __name__ == "__main__":
    main()
