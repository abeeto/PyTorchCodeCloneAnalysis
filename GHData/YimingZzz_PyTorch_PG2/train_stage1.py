import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io

from unet_generator_stage1 import *
from data_loading import *

#prepare the dataset
train_list = []
with open('/home/yiming/code/data/list_train.json','r') as f:
    new_list = f.read().split('\n')[:-1]
    for x in new_list:
        json_x = json.loads(x)
        train_list.append(json_x)
f.close

train_pair_list = []
group_id_list = []
counter = 0
for i in range(len(train_list)):
    names = train_list[i].split('_')
    img_id = names[0]
    if i == 0:
        group_id_list.append([train_list[i]])
    else:
        if img_id == train_list[i-1].split('_')[0]:
            group_id_list[counter].append(train_list[i])
        else:
            counter += 1
            group_id_list.append([train_list[i]])            
#print(len(group_id_list))
for item in group_id_list:
    for i in range(len(item)):
        for j in range (i+1, len(item)):
            train_pair_list.append([item[i], item[j]])

train_flip_list = []
with open('/home/yiming/code/data/list_train_flip.json','r') as f:
    new_list = f.read().split('\n')[:-1]
    for x in new_list:
        json_x = json.loads(x)
        train_flip_list.append(json_x)
f.close
#print (len(train_flip_list))

train_flip_pair_list = []
group_flip_id_list = []
counter = 0
for i in range(len(train_flip_list)):
    names = train_flip_list[i].split('_')
    img_id = names[0]
    if i == 0:
        group_flip_id_list.append([train_flip_list[i]])
    else:
        if img_id == train_flip_list[i-1].split('_')[0]:
            group_flip_id_list[counter].append(train_flip_list[i])
        else:
            counter += 1
            group_flip_id_list.append([train_flip_list[i]])            

#print (len(group_flip_id_list))
for item in group_flip_id_list:
    for i in range(len(item)):
        for j in range (i+1, len(item)):
            train_flip_pair_list.append([item[i], item[j]])
            
train_pair_list = train_pair_list + train_flip_pair_list

test_list = []
with open('/home/yiming/code/data/list_test_samples.json', 'r') as f:
    new_list = f.read().split('\n')[:-1]
    for x in new_list:
        json_x = json.loads(x)
        test_list.append(json_x)
f.close

group_test_id_list = []
test_pair_list = []    
counter = 0
for i in range(len(test_list)):
    names = test_list[i].split('_')
    img_id = names[0]
    if i == 0:
        group_test_id_list.append([test_list[i]])
    else:
        if img_id == test_list[i-1].split('_')[0]:
            group_test_id_list[counter].append(test_list[i])
        else:
            counter += 1
            group_test_id_list.append([test_list[i]])            

#print (len(group_flip_id_list))
for item in group_test_id_list:
    for i in range(len(item)):
        for j in range (i+1, len(item)):
            test_pair_list.append([item[i], item[j]])




DF_dataset_train = DeepFashionDataset(file_list = train_pair_list, mode = "train")
dataloader_train = DataLoader(DF_dataset_train, batch_size = 4, shuffle = True, num_workers = 4)
DF_dataset_test = DeepFashionDataset(file_list = test_pair_list, mode = "test")
dataloader_test = DataLoader(DF_dataset_test, batch_size = 1, shuffle = False, num_workers = 4)

#set cuda for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize the network
U_stage1 = UnetGenerator()
U_stage1.to(device)

#set criterion and optimizer
criterion = nn.L1Loss().to(device)
optimizerL1 = torch.optim.Adam(U_stage1.parameters(), lr = 0.0001, betas = (0.5, 0.9))

#parameters for training
#train_iterations = []
train_loss = []
epoch_num = 2
test_id = 1
for epoch in range(epoch_num):
    print ("epoch %d" % epoch)
    for i_train, train_sample in enumerate(dataloader_train):
        #print (i_batch)
        running_loss = 0.0
        #get inputs and groundtruth
        source_img = train_sample['source_img']
        target_pose = train_sample['target_pose']
        inputs = torch.cat([source_img, target_pose], dim = 1).to(device)
        target_img = train_sample['target_img'].to(device)
        outputs = U_stage1(inputs)
        #computen loss and backprop
        loss = criterion(outputs, target_img)
        #print (loss.data.item())
        #reset optimizer
        optimizerL1.zero_grad()
        loss.backward()
        optimizerL1.step()

        running_loss = loss.data.item()

        if (i_train == 0) or (i_train+1) % 500 == 0:
            print("iteration: %d"%(i_train+1))

            print ("loss: %f" % running_loss)
            for i_test, test_sample in enumerate(dataloader_test):
                print (i_test, test_sample['source_img'].size(), test_sample['target_img'].size(), test_sample['target_pose'].size())
                source_img = test_sample['source_img'].numpy()
                source_img = np.squeeze(source_img)
                source_img = source_img.transpose([1, 2, 0])    
                io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_test/source_img' + str(test_id) + '.jpg', source_img)

                target_img = test_sample['target_img'].numpy()
                target_img = np.squeeze(target_img)
                target_img = target_img.transpose([1, 2, 0])
                io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_test/target_img' + str(test_id) + '.jpg', target_img)

                inputs = torch.cat([test_sample['source_img'], test_sample['target_pose']], dim = 1).to(device)
                outputs = U_stage1(inputs).cpu()
                #a.requires_grad_(True)
                outputs_no_grad = outputs.detach()
                generated_result = (np.squeeze(outputs_no_grad.numpy())).transpose([1, 2, 0])
                io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_test/generated_result' + str(test_id) + '.jpg', generated_result)
                if i_test == 0:
                    print('result saved\n')
                    test_id += 1
                    break
        train_loss.append(running_loss)        





'''
for epoch in range(epoch_num):
    print ("epoch %d" % epoch)
    for i_batch, sample_batch in enumerate(dataloader):
        print (i_batch)
        running_loss = 0.0
        #get inputs and groundtruth
        source_img = sample_batch['source_img']
        target_pose = sample_batch['target_pose']
        inputs = torch.cat([source_img, target_pose], dim = 1).to(device)
        target_img = sample_batch['target_img'].to(device)
        outputs = U_stage1(inputs)
        #computen loss and backprop
        loss = criterion(outputs, target_img)
        print (loss.data.item())
        #reset optimizer
        optimizerL1.zero_grad()
        loss.backward()
        optimizerL1.step()

        running_loss = loss.data.item()

        if (i_batch+1) % 100 == 0:
            print ("loss: %f" % running_loss)
        


        train_loss.append(running_loss)
'''