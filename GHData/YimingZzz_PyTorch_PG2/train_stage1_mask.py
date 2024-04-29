import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.measure import compare_ssim as ssim

from unet_generator_stage1 import *
from data_loading import *

model_path = '/media/qiujing/91ec90ab-87ac-41f3-9f23-fbbbf9c36c61/U_model/stage1_mask/'



def list2json(list_name, json_path, json_name):
    json_file = open(os.path.join(json_path, json_name), 'a')
    for i in list_name:
        json_i = json.dumps(i)
        json_file.write(json_i + '\n')
    json_file.close()

def json2list(json_path, json_name):
    read_list = []
    with open(os.path.join(json_path, json_name), 'r') as f:
        new_list = f.read().split('\n')[:-1]
        for x in new_list:
            json_x = json.loads(x)
            read_list.append(json_x)
    f.close
    return read_list

train_list = json2list('/home/yiming/code/data/', 'list_train.json')

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


train_flip_list = json2list('/home/yiming/code/data/', 'list_train_flip.json')

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

test_list = json2list('/home/yiming/code/data', 'list_test_samples.json')
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize the network
U_stage1 = UnetGenerator()
U_stage1.to(device)

#set criterion and optimizer
criterion = nn.L1Loss().to(device)
optimizerL1 = torch.optim.Adam(U_stage1.parameters(), lr = 0.00002, betas = (0.5, 0.999))

#parameters for training
#train_iterations = []
train_loss_record = []
train_l1_loss_record = []
train_mask_loss_record = []
ssim_list = []
train_l1_loss = 0.0
train_mask_loss = 0.0
train_loss = 0.0
epoch_num = 5
test_id = 1
batch_size = 4

for epoch in range(epoch_num):
    print ("epoch %d" % (epoch+1))
    for i_train, train_sample in enumerate(dataloader_train):
        #print (i_batch)
        pose_mask = train_sample['pose_mask']
        pose_mask = torch.reshape(pose_mask, [batch_size, 1, 256, 256])
        pose_mask1 = pose_mask
        pose_mask2 = pose_mask
        pose_mask3 = pose_mask
        pose_mask = torch.cat((pose_mask1, pose_mask2, pose_mask3), dim = 1).to(device)
                #get inputs and groundtruth
        source_img = train_sample['source_img'].to(device)
        target_pose = train_sample['target_pose'].to(device)
        #inputs = torch.cat([source_img, target_pose], dim = 1).to(device)
        target_img = train_sample['target_img'].to(device)
        target_img_mask = target_img * pose_mask

        #outputs = U_stage1(inputs)
        outputs = U_stage1(source_img, target_pose)
        outputs_mask = outputs * pose_mask

        #computen loss and backprop
        loss1 = criterion(outputs, target_img)
        #print (loss1.data.item())
        loss2 = criterion(outputs_mask, target_img_mask)
        #print (loss2.data.item())
        loss = loss1 + loss2
        #print(loss.data.item())
        #print (loss.data.item())
        #reset optimizer
        optimizerL1.zero_grad()
        loss.backward()
        optimizerL1.step()

        train_l1_loss += loss1.data.item()
        train_mask_loss += loss2.data.item()
        train_loss += loss.data.item()

        if (i_train == 0) or (i_train+1) % 100 == 0:
            print("iteration: %d"%(i_train+1))
            if i_train == 0:
                train_l1_loss = train_l1_loss
                train_mask_loss = train_mask_loss
                train_loss = train_loss

            else:
                train_l1_loss = train_l1_loss / 100
                train_mask_loss = train_mask_loss / 100
                train_loss = train_loss / 100           

            train_l1_loss_record.append(train_l1_loss)
            train_mask_loss_record.append(train_mask_loss)
            train_loss_record.append(train_loss)

            print ("train_l1_loss: %f" % train_l1_loss)
            print ("train_mask_loss: %f" % train_mask_loss)                
            print ("train_loss: %f" % train_loss)
            train_l1_loss = 0.0
            train_mask_loss = 0.0
            train_loss = 0.0

        if (i_train + 1)%1000 == 0:
            #save model
            net_param_id = int((i_train + 1)/ 1000)
            torch.save(U_stage1.state_dict(), os.path.join(model_path, 'U_stage1_params' + '_' + str(epoch+1) + '_' + str(net_param_id) + '.pkl'))
                        
            #visualize some results
            for i_test, test_sample in enumerate(dataloader_test):
                #print (i_test, test_sample['source_img'].size(), test_sample['target_img'].size(), test_sample['target_pose'].size())
                if (i_test + 1) % 30 == 0:
                    test_num = int((i_test + 1)/30)
                    print ('test %d'%test_num)

                    source_img = test_sample['source_img'].numpy()
                    source_img = np.squeeze(source_img)
                    source_img = source_img.transpose([1, 2, 0])    
                    if epoch == 0 and i_train == 999:
                        io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_mask/source_img' + str(test_num) + '.jpg', source_img)

                    target_img = test_sample['target_img'].numpy()
                    target_img = np.squeeze(target_img)
                    target_img = target_img.transpose([1, 2, 0])
                    if epoch == 0 and i_train == 999:
                        io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_mask/target_img' + str(test_num) + '.jpg', target_img)
                
                    #inputs = torch.cat([test_sample['source_img'], test_sample['target_pose']], dim = 1).to(device)
                    outputs = U_stage1(test_sample['source_img'].to(device), test_sample['target_pose'].to(device)).cpu()
                    outputs_no_grad = outputs.detach()
                    generated_result = (np.squeeze(outputs_no_grad.numpy())).transpose([1, 2, 0])
                    io.imsave('/home/yiming/code/data/DeepFashion/DF_img_pose/stage1_mask/generated_result' + str(test_num) + '_iter_' + str(test_id*1000) + '.jpg', generated_result)
            
            print('results saved\n')
            test_id += 1

            #record ssim and loss on test data    
            for i_test, test_sample in enumerate(dataloader_test):
                #ssim_result = 0.0

                target_img = test_sample['target_img'].numpy()
                target_img = np.squeeze(target_img)
                target_img = target_img.transpose([1, 2, 0])

                #inputs = torch.cat([test_sample['source_img'], test_sample['target_pose']], dim = 1).to(device)
                #outputs = U_stage1(inputs).cpu()
                outputs = U_stage1(test_sample['source_img'].to(device), test_sample['target_pose'].to(device)).cpu()

                outputs_no_grad = outputs.detach()
                generated_result = (np.squeeze(outputs_no_grad.numpy())).transpose([1, 2, 0])

                target_img_gray = rgb2gray(target_img)
                generated_result_gray = rgb2gray(generated_result)

                ssim_result += ssim(generated_result_gray, target_img_gray, data_range = target_img_gray.max() - target_img_gray.min())

            ssim_result /= (i_test + 1)
            ssim_list.append(ssim_result)
            ssim_result = 0.0
            #save record
            list2json(train_l1_loss, '/home/yiming/code/data/record/stage1_mask', 'train_l1_loss_record.json')
            train_l1_loss_record = []
            list2json(train_mask_loss, '/home/yiming/code/data/record/stage1_mask', 'train_mask_loss_record.json')
            train_mask_loss_record = []
            list2json(train_loss, '/home/yiming/code/data/record/stage1_mask', 'train_loss_record.json')
            train_loss_record = []
            list2json(ssim_list, '/home/yiming/code/data/record/stage1_mask', 'ssim_record.json')
            ssim_list = []