import torch.nn as nn
import torch
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from data_loader import dataloader
import pandas as pd
import json
from data_loader import transform
from models import classification as cls
import numpy as np
from torchvision import transforms
import PIL
# use this file if you want to quickly test your model


def tta_labels(model,test_loader,device,cfg):
    # scaler = StandardScaler()
    # clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    model.eval()
    model = model.to(device)
    test_features = np.empty((256,))
    all_preds = []
    all_labels = []
    for images,labels in test_loader:
        # for each image in the test set
        replicated = []
        for image in images:
            image = transforms.ToPILImage()(image)
            image_list = [
                          transforms.RandomVerticalFlip(p=1)(image),
                          transforms.ColorJitter(hue=.05, saturation=.05)(image),
                          transforms.RandomPerspective(p=1)(image),
                          transforms.RandomAffine(degrees=20)(image),
                          transforms.RandomRotation(90, resample=PIL.Image.BILINEAR)(image),
                          transforms.RandomHorizontalFlip(p=1)(image),
            ]
            # turn all replications into tensors
            replications = [transforms.ToTensor()(image) for image in image_list
                            
            ]
            replicated = torch.stack([r for r in replications])
            # predict all both models
            replicated = replicated.to(device)
            # preds = model(replicated).cpu().detach().numpy()
            with torch.no_grad():
                output = model(replicated)
                output_sigmoid = torch.sigmoid(output)               
                preds = (output_sigmoid.cpu().numpy() >0.5).astype(float)
            # print(len(labels))
        
            all_preds.extend([np.mean(preds,axis=0)>0.5])
        all_labels.extend(labels.tolist())
    return (classification_report(all_labels, all_preds, target_names=cfg["data"]["label_dict"]))
    

def test_result(model, test_loader, device, cfg):
    # testing the model by turning model "eval" mode
    model.eval()
    list_labels = []
    list_preds = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            outputs_softmax = torch.softmax(outputs, dim=-1)
        
        probs, preds = torch.max(outputs_softmax.data, dim=-1)
        list_labels.extend(targets.cpu().detach().numpy())
        list_preds.extend(preds.cpu().detach().numpy())

    return (classification_report(list_labels, list_preds, target_names=cfg["data"]["label_dict"]))

def adaptive_test_result(model, test_loader, device,cfg,num_of_class=2):
    # testing the model by turning model "Eval" mode
    model.eval()
    list_pres = []
    list_label = []
    list_sigmoid = []
    # preds = []
    labels = []
    to_write = open('log_test.txt',"w+")
    for data, abnormal, target in test_loader:
        # data = data.to(device)
        target = target.to(device).long()
        # remember to unlock for 2 branch
        # abnormal = abnormal.to(device)
        bs = len(data)

        with torch.no_grad():
            Y_pred = torch.empty((0,2)).to(device)
            for i in range(0,bs):
                input_ecg = data[i].unsqueeze(0).to(device)
                abnormal_3 = abnormal[i].unsqueeze(0).to(device)
                # print(input_ecg.shape)
                # print("shape:",abnormal.shape)
                # print(abnormal[[i]].unsqueeze(0).shape)
                preds = model(input_ecg,abnormal_3)
                # preds = model(input_ecg)
                # preds = model(input_ecg,abnormal[[i]].unsqueeze(0).unsqueeze(0))
                Y_pred = torch.cat((Y_pred,preds))
            preds = torch.softmax(Y_pred,dim=-1).cpu().detach().numpy()
            preds = np.argmax(preds,axis=-1)
            # preds = (output_sigmoid.cpu().numpy() >0.05).astype(float)
 
            to_write.write('{},{}\n'.format(preds.tolist(),target.tolist()))
        list_label.extend(target.tolist())  
        list_pres.extend(preds.tolist())  
    to_write.close()
    print(len(list_label))
    print(len(list_pres))
    return (classification_report(list_label, list_pres, digits=4,target_names=cfg["data"]["label_dict"]))


    print("Testing process beginning here....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tenes.cfg', help='JSON file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    print("Testing process beginning here....")
    # read configure file
    with open(args.configure) as f:
        cfg = json.load(f)

    # load model
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    test_model = checkpoint['model']
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)

    print("Inference on the testing set")
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data)
    # prepare the dataset
    testing_set = dataloader.ClassificationDataset(test_df, data_path, transform.val_transform)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False,)
    print(test_result(model, test_loader, device,cfg))
    # print(tta_labels(model,test_loader,device,cfg))