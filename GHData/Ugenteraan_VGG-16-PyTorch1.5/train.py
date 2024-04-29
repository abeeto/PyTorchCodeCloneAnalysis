'''
Training script for VGG-16.
'''
from tqdm import tqdm
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from image_transforms import RandomRotate, RandomHorizontalFlip, RandomNoise, RandomVerticalFlip, RandomShear, ToTensor
from model import Model
from load_data import LoadDataset
from utils import calculate_accuracy, plot_graph
import train_cfg as t_cfg

def main():
    '''
    Train function.
    '''

    ########################################################## Model Initialization & Loading ##########################################################
    vgg = Model(resized_img_size=t_cfg.RESIZED_IMAGE_SIZE, num_classes=t_cfg.NUM_CLASSES, init_weights=True)

    optimizer = Adam(vgg.parameters(), lr=t_cfg.LEARNING_RATE) #optimizer
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=t_cfg.LR_DECAY_RATE) #scheduler is used to lower the learning rate during training later.
    loss_criterion = torch.nn.CrossEntropyLoss() #loss function.

    vgg = vgg.to(t_cfg.DEVICE) #move the network to GPU if available.

    print("--- Model Architecture ---")
    print(vgg)

    if t_cfg.TRAINED_MODEL_PRESENCE:

        model_params = torch.load(t_cfg.MODEL_PATH+t_cfg.MODEL_NAME) #reads parameters from the model file.
        vgg.load_state_dict(model_params) #load the parameters into the model architecture.
        print("Model parameters are loaded from the saved file!")



    ########################################################## Data Initialization & Loading ##########################################################
    #Initialize the training data class.
    training_data = LoadDataset(resized_image_size=t_cfg.RESIZED_IMAGE_SIZE, total_images=t_cfg.TOTAL_TRAIN_DATA, classes=t_cfg.CLASSES,
                                data_list=t_cfg.TRAIN_IMG_LABEL_LIST, transform=transforms.Compose([RandomRotate(angle_range=t_cfg.ROTATION_RANGE, prob=t_cfg.ROTATION_PROB),
                                                                                                    RandomShear(shear_range=t_cfg.SHEAR_RANGE, prob=t_cfg.SHEAR_PROB),
                                                                                                    RandomHorizontalFlip(prob=t_cfg.HFLIP_PROB),
                                                                                                    RandomVerticalFlip(prob=t_cfg.VFLIP_PROB),
                                                                                                    RandomNoise(mode=t_cfg.NOISE_MODE, prob=t_cfg.NOISE_PROB),
                                                                                                    ToTensor(mode=0)]))

    testing_data = LoadDataset(resized_image_size=t_cfg.RESIZED_IMAGE_SIZE, total_images=t_cfg.TOTAL_TEST_DATA, classes=t_cfg.CLASSES,
                                data_list=t_cfg.TEST_IMG_LABEL_LIST, transform=transforms.Compose([ToTensor(mode=0)]))

    train_dataloader = DataLoader(training_data, batch_size=t_cfg.BATCH_SIZE, shuffle=t_cfg.DATA_SHUFFLE, num_workers=t_cfg.NUM_WORKERS)
    test_dataloader = DataLoader(testing_data, batch_size=t_cfg.BATCH_SIZE, shuffle=t_cfg.DATA_SHUFFLE, num_workers=t_cfg.NUM_WORKERS)




    ########################################################## Model Training & Saving ##########################################################
    best_accuracy = 0

    entire_loss_list = []
    entire_accuracy_list = []

    for epoch_idx in range(t_cfg.EPOCH):

        print("Training for epoch %d has started!"%(epoch_idx+1))

        vgg.train()
        epoch_training_loss = []
        epoch_accuracy = []
        i = 0
        for i, sample in tqdm(enumerate(train_dataloader)):

            batch_x, batch_y = sample['image'].to(t_cfg.DEVICE), sample['label'].to(t_cfg.DEVICE)

            optimizer.zero_grad() #clear the gradients in the optimizer between every batch.

            net_output = vgg(batch_x) #output from the network.

            total_loss = loss_criterion(input=net_output, target=batch_y)

            epoch_training_loss.append(total_loss.item()) #append the loss of every batch.

            total_loss.backward() #calculate the gradients.
            optimizer.step()

            batch_acc = calculate_accuracy(network_output=net_output, target=batch_y)
            epoch_accuracy.append(batch_acc.cpu().numpy())

        lr_decay.step() #decay rate update
        curr_accuracy = sum(epoch_accuracy)/(i+1)
        curr_loss = sum(epoch_training_loss)

        print("The accuracy at epoch %d is %g"%(epoch_idx, curr_accuracy))
        print("The loss at epoch %d is %g"%(epoch_idx, curr_loss))

        entire_accuracy_list.append(curr_accuracy)
        entire_loss_list.append(curr_loss)

        vgg.eval()
        epoch_testing_accuracy = []
        epoch_testing_loss = []
        j = 0
        with torch.no_grad():

            for j, test_sample in tqdm(enumerate(test_dataloader)):

                batch_x, batch_y = test_sample['image'].to(t_cfg.DEVICE), test_sample['label'].to(t_cfg.DEVICE)

                net_output = vgg(batch_x)

                total_loss = loss_criterion(input=net_output, target=batch_y)

                epoch_testing_loss.append(total_loss.item())

                batch_acc = calculate_accuracy(network_output=net_output, target=batch_y)
                epoch_testing_accuracy.append(batch_acc.cpu().numpy())

            test_accuracy = sum(epoch_testing_accuracy)/(j+1)
            test_loss = sum(epoch_testing_loss)

            print("The testing accuracy at epoch %d is %g"%(epoch_idx, test_accuracy))
            print("The testing loss at epoch %d is %g"%(epoch_idx, test_loss))


        if test_accuracy > best_accuracy:

            torch.save(vgg, t_cfg.MODEL_PATH + t_cfg.MODEL_NAME)
            best_accuracy = test_accuracy
            print("Model is saved !")

    ########################################################## Graphs ##########################################################
    if t_cfg.PLOT_GRAPH:
        plot_graph(t_cfg.EPOCH, "Epoch", "Training Loss", "Training Loss for %d epoch"%(t_cfg.EPOCH), "./loss.png", [entire_loss_list, 'r--', "Loss"])
        plot_graph(t_cfg.EPOCH, "Epoch", "Training Accuracy", "Training Accuracy for %d epoch"%(t_cfg.EPOCH), "./accuracy.png", [entire_accuracy_list, 'b--', "Accuracy"])



if __name__ == "__main__":
    main()
