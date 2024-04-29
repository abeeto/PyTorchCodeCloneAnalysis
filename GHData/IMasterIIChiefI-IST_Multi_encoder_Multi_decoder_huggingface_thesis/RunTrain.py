import json
import os, sys
import time
from tkinter import filedialog
from typing import List
import torch
import torch.cuda
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, BeitFeatureExtractor, DeiTFeatureExtractor, \
    RobertaTokenizer, GPT2Tokenizer
import random
import numpy as np
from datetime import datetime
import re
from custom_modeling_Dual_encoder_Multiple_decoder import VisionEncoderDecoderModel
print("cache_cleaner")
torch.cuda.empty_cache()

print("dataset_joined file")
DataSetPath = 'E:/Users/Halo_/Desktop/thesis/dataset/Dataset_latest_not/Transformer_DataSet/Images_Filtered_Cropped/Dataset.Json'
FakeDataSetPath = 'E:/Users/Halo_/Desktop/thesis/dataset/Dataset_latest_not_not/Transformer_DataSet/Images_Filtered_Cropped/FakeDataset.Json'
Validation = False
#EncoderName = "google/vit-base-patch16-224-in21k" #default
EncoderName = "google/vit-large-patch32-384" #default
#EncoderName = "microsoft/beit-base-patch16-224" #didnt work with pretrained
#EncoderName = "microsoft/beit-base-patch16-224-pt22k-ft22k" #didnt work with pretrained
#EncoderName = "facebook/deit-base-distilled-patch16-224" #didnt work with pretrained
DecoderName = "gpt2" #default
#DecoderName = "gpt2-large" #not having memory enough 
#DecoderName = "distilgpt2" #similiar results as gpt2 smaller memory footprint
#DecoderName = "roberta-base" #didnt work with pretrained
#DecoderName = "roberta-large" #since the roberta base didnt brought result we dont tested large variant
#DecoderName = "EleutherAI/gpt-j-6B" #too large model to be pratical

#------------------------------------------------------------------------------------------------OPTIONS
now = datetime.now()
dtString = now.strftime("%d-%m-%Y")
SaveCheckPointsPath = './' + EncoderName.capitalize().replace("/", "-").replace(" ", "_") + 'to' + DecoderName.capitalize().replace("/", "_").replace(" ", "_") + "_" + dtString
batch_size = 1
num_epochs = 170
noise_epochs = 2
save_periodically_epochs = 10
MaxCheckPoint = int(num_epochs/save_periodically_epochs)+1
learningrate = 1e-7 #1e-28
OPTIM = "adamw"
#------------------------------------------------------------------------------------------------OPTIONS

def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def Retrieve_Epoch(Str):
    return int(re.search("Epoch_(\d+)", Str).group(1))


def CollateCustom(batch):
    Sizes = []
    for item in batch:
        Sizes.append(len(item[4])+1)
    YName = []
    YLocation = []
    YDate = []
    XFullImage = []
    XCrop = np.empty((len(batch), max(Sizes)), dtype=object)
    for i in range(len(batch)):
        YName.append(batch[i][0])
        YLocation.append(batch[i][1])
        YDate.append(batch[i][2])
        XFullImage.append(batch[i][3])
        XCrop[i][0:len(batch[i][4])] = batch[i][4]
    return YName, YLocation, YDate, XFullImage, XCrop, Sizes


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataSetPath, EncoderName, DecoderName):
        super(CustomDataset).__init__()
        try:
            with open(dataSetPath) as dataSetPath_crop:
                data = json.load(dataSetPath_crop)
                self.YData = {
                    'Name/Family': data["YData"]['Name/Family'],
                    'Location': data["YData"]['Location'],
                    'Date': data["YData"]['Date']
                }
                self.XPath = {
                    'Crop': data["XPath"]['Crop'],
                    'FullImage': data["XPath"]['FullImage']
                }
                self.Numbers = data["YData"]["Number"]
        except:
            dataSetPath = filedialog.askopenfilename(title="Select Dataset File",
                                                          filetypes=[('Checkpoint', '*.pth')])
            with open(dataSetPath) as dataSetPath_crop:
                data = json.load(dataSetPath_crop)
                self.YData = {
                    'Name/Family': data["YData"]['Name/Family'],
                    'Location': data["YData"]['Location'],
                    'Date': data["YData"]['Date']
                }
                self.XPath = {
                    'Crop': data["XPath"]['Crop'],
                    'FullImage': data["XPath"]['FullImage']
                }
                self.Numbers = data["YData"]["Number"]
        if "vit" in EncoderName:
            self.encTokens = ViTFeatureExtractor.from_pretrained(EncoderName)
        if "beit" in EncoderName:
            self.encTokens = BeitFeatureExtractor.from_pretrained(EncoderName)
        if "deit" in EncoderName:
            self.encTokens = DeiTFeatureExtractor.from_pretrained(EncoderName)
        if "roberta" in DecoderName:
            self.decTokens = RobertaTokenizer.from_pretrained(DecoderName)
        if "gpt2" in DecoderName:
            self.decTokens = GPT2Tokenizer.from_pretrained(DecoderName)

        self.decTokens.eos_token = model.config.eos_token_id
        self.decTokens.pad_token = model.config.pad_token_id
        self.decTokens.add_special_tokens = True

    def __getitem__(self, idx):
        Name: str
        Location: str
        Date: str
        Crop: Image
        FullImage: Image
        while len(self.YData['Name/Family'][idx][0]) < 3 or len(self.YData['Location'][idx][0]) < 3 or len(self.YData['Date'][idx][0]) < 3:
            idx = random.randint(0, self.__len__())

        try:
            Name = (self.YData['Name/Family'][idx][0] + ", " + self.YData['Name/Family'][idx][1])
        except:
            Name = (self.YData['Name/Family'][idx][0])

        try:
            Location = (self.YData['Location'][idx][0] + ", " + self.YData['Location'][idx][1])
        except:
            Location = (self.YData['Location'][idx][0])

        Date = (self.YData['Date'][idx][0])

        FullImage = Image.open(self.XPath['FullImage'][idx])

        Crops = []
        for Crop in self.XPath['Crop'][idx]:
            Crops.append(Image.open(Crop).convert('YCbCr').getchannel(0).convert('RGB'))

        Values = []
        for Crop in Crops:
            CropWidth, CropHeight = Crop.size
            Values.append(CropWidth)
            Values.append(CropHeight)

        ImageWidth, ImageHeight = FullImage.size
        ImageRatio = ImageWidth/ImageHeight
        FullImage = FullImage.resize((max(Values), int(max(Values)/ImageRatio)), Image.ANTIALIAS)
        return Name, Location, Date, FullImage, Crops,

    def __len__(self):
        return len(self.XPath['FullImage'])

    def decode(self, logits, skip_special_tokens=True, clean_up_tokenization_spaces=True, spaces_between_special_tokens=True):
        Output_text = []
        token_ids = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        for token_id in token_ids:
            filtered_tokens = self.decTokens.convert_ids_to_tokens(token_id, skip_special_tokens=skip_special_tokens)
            sub_texts = []
            current_sub_text = []
            for token in filtered_tokens:
                if skip_special_tokens and token in self.decTokens.all_special_ids:
                    continue
                if token in self.decTokens.added_tokens_encoder:
                    if current_sub_text:
                        sub_texts.append(self.decTokens.convert_tokens_to_string(current_sub_text))
                        current_sub_text = []
                    sub_texts.append(token)
                else:
                    current_sub_text.append(token)
            if current_sub_text:
                sub_texts.append(self.decTokens.convert_tokens_to_string(current_sub_text))

            if spaces_between_special_tokens:
                text = " ".join(sub_texts)
            else:
                text = "".join(sub_texts)
            if clean_up_tokenization_spaces:
                text = self.decTokens.clean_up_tokenization(text)
            Output_text.append(text)
        return Output_text

try:
    os.mkdir(SaveCheckPointsPath)
except OSError as error:
    print('Folder creation error or already exists ')

try:
    model = VisionEncoderDecoderModel.from_pretrained(SaveCheckPointsPath, training=True)
    print("config found on folder will be loaded this potentially ignore different initialization parameters")
except:
    print("initialization a New Model")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_pretrained_model_name_or_path=EncoderName, decoder_pretrained_model_name_or_path=DecoderName, training=True)
    model.config.bos_token_id = model.decoder.config.eos_token_id
    model.config.eos_token_id = model.decoder.config.eos_token_id
    model.config.decoder_start_token_id = model.config.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.save_pretrained(SaveCheckPointsPath)


Listing = os.listdir(SaveCheckPointsPath)
Listing: List[str] = list(filter(lambda elem: '.pth' in elem, Listing))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#------------------------------------------------------------------------------------------------OPTIMIZERS
if "adamw" == OPTIM:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningrate)
elif "adam" == OPTIM:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
elif "sgd" == OPTIM:
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.949)
else:
    print("optimizer not implemented")
    exit(-1)
#------------------------------------------------------------------------------------------------OPTIMIZERS

if len(Listing) >= 1:
    CheckpointPath = filedialog.askopenfilename(initialdir=os.path.normpath(SaveCheckPointsPath))
    if CheckpointPath is not None:
        '''
        #FASTER BUT LESS GPU MEMORY EFFICIENT WHEN LOADING
            checkpoint = torch.load(CheckpointPath)
            model.load_state_dict(checkpoint['model_state_dict'])
            if not Validation:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            head_tail = os.path.split(CheckpointPath)
            epochCheckpointoffset = Retrieve_Epoch(head_tail[1])
        '''
        #SLOWER BUT USES MORE RAM AND LESS GPU MEMORY WHEN LOADING
        from collections import OrderedDict as OD
        checkpoint = torch.load(CheckpointPath, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        encoder_checkpoint = {argument[len("encoder."):]: value for argument, value in model_state_dict.items() if
                              argument.startswith("encoder.")}
        enc_to_dec_proj_checkpoint = {argument[len("enc_to_dec_proj."):]: value for argument, value in model_state_dict.items() if
                                      argument.startswith("enc_to_dec_proj")}
        decoder_checkpoint = {argument[len("decoder."):]: value for argument, value in model_state_dict.items() if
                              argument.startswith("decoder.")}
        encoder_state_dict = OD()
        encoder_state_dict.update(encoder_checkpoint)
        model.encoder.load_state_dict(encoder_state_dict)
        if len(enc_to_dec_proj_checkpoint) > 0:
            enc_to_dec_proj_state_dict = OD()
            enc_to_dec_proj_state_dict.update(enc_to_dec_proj_checkpoint)
            model.enc_to_dec_proj.load_state_dict(enc_to_dec_proj_state_dict)
        decoder_state_dict = OD()
        decoder_state_dict.update(decoder_checkpoint)
        model.decoder.load_state_dict(decoder_state_dict)
        model = model.to(device)
        torch.cuda.empty_cache()
        if not Validation:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(optimizer, device)
            torch.cuda.empty_cache()
        head_tail = os.path.split(CheckpointPath)
        epochCheckpointoffset = Retrieve_Epoch(head_tail[1])
    else:
        epochCheckpointoffset = 0
else:
    epochCheckpointoffset = 0

FilePath = [None] * MaxCheckPoint
dataset = CustomDataset(DataSetPath, EncoderName=EncoderName, DecoderName=DecoderName)
fakedataset = CustomDataset(FakeDataSetPath, EncoderName=EncoderName, DecoderName=DecoderName)


if epochCheckpointoffset < num_epochs:
    num_epochs -= epochCheckpointoffset

num_epochs = int(num_epochs)
num_noise_epochs = list(range(0, num_epochs, noise_epochs))
save_epochs = list(range(0, num_epochs, save_periodically_epochs))

#our validation dataset is already diferent from the training dataset from the filtered files the via go for training and the rest to validation
#train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7), int(len(dataset)*0.5)])
#train_fakedataset, val_fakedataset = torch.utils.data.random_split(fakedataset, [int(len(fakedataset)*0.7), int(len(fakedataset)*0.5)])
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateCustom)
#faketrain_dataloader = DataLoader(train_fakedataset, batch_size=batch_size, shuffle=True, collate_fn=CollateCustom)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateCustom)
faketrain_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateCustom)
#faketrain_dataloader = DataLoader(fakedataset, batch_size=batch_size, shuffle=True, collate_fn=CollateCustom)
#------------------------------------------------------------------------------------------------scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#scheduler2 = StepLR(optimizer, step_size=10, gamma=0.1)
#------------------------------------------------------------------------------------------------scheduler

if not Validation:
    model.train()
    model.training = True
    n_iters = 0
    for epoch in range(num_epochs):
        for i, (YName, YLocation, YDate, XFullImage, XCrop, Sizes) in enumerate(train_dataloader if epoch in num_noise_epochs else faketrain_dataloader):
            optimizer.zero_grad()
            Images_p = torch.zeros((len(Sizes), max(Sizes)*3, dataset.encTokens.size, dataset.encTokens.size), dtype=torch.float32)
            for i in range(len(Sizes)):
                Images = dataset.encTokens(images=[XFullImage[i], *XCrop[i][0:Sizes[i]-1].tolist()], return_tensors="pt")
                Images_p[i][0:3 * Sizes[i]][:][:] = Images["pixel_values"].reshape(1, 3 * Sizes[i], dataset.encTokens.size, dataset.encTokens.size)
            Labels = dataset.decTokens([*YName, *YLocation, *YDate], padding=True, truncation=True, max_length=512, return_tensors="pt")
            output = model(pixel_values=Images_p.to(device),
                           labels=Labels["input_ids"].to(device),
                           decoder_attention_mask=Labels["attention_mask"].to(device),
                           single_loss_sum=True,
                           encoder_batch_lengths=Sizes)
            if output.loss is not None:
                loss = output.loss
                loss.backward()
                if output.loss_name is not None and output.loss_location is not None and output.loss_date is not None:
                    loss_name = output.loss_name
                    loss_location = output.loss_location
                    loss_date = output.loss_date
                else:
                    loss_name = loss
                    loss_location = loss
                    loss_date = loss
            elif output.loss_name is not None and output.loss_location is not None and output.loss_date is not None:
                loss_name = output.loss_name
                loss_location = output.loss_location
                loss_date = output.loss_date
                loss_date.backward(retain_graph=True)
                loss_location.backward(retain_graph=True)
                loss_name.backward()
            else:
                raise ValueError("No Labels were provided")
            if n_iters % 6 == 0:
                clear()
                text = dataset.decode(output.decoder_outputs.logits_name)
                print(50 * "-" + "Name" + (200-len("Name")) * "-")
                print("Labels originals:")
                print(YName)
                print("Labels Prediction:")
                print(text)
                print("\n Training loss =" + str(loss_name.data.item()) + "|| epoch =" + str(
                    epoch + epochCheckpointoffset) + "/" + str(num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                    n_iters))
                print(50 * "-" + "end" + (200-len("end")) * "-")

                text = dataset.decode(output.decoder_outputs.logits_location)
                print(50 * "*" + "Begin_Location" + (200-len("Begin_Location")) * "*")
                print("Labels originals:")
                print(YLocation)
                print("Labels Prediction:")
                print(text)
                print("\n Training loss =" + str(loss_location.data.item()) + "|| epoch =" + str(
                    epoch + epochCheckpointoffset) + "/" + str(num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                    n_iters))
                print(50 * "*" + "end" + (200-len("end")) * "*")

                text = dataset.decode(output.decoder_outputs.logits_date)
                print(50 * "+" + "Begin_Date" + (200-len("Begin_Date")) * "+")
                print("Labels originals:")
                print(YDate)
                print("Labels Prediction:")
                print(text)
                print("\n Training loss =" + str(loss_date.data.item()) + "|| epoch =" + str(
                    epoch + epochCheckpointoffset) + "/" + str(num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                    n_iters))
                print(50 * "+" + "end" + (200-len("end")) * "+")
                del output
            optimizer.step()
            n_iters += batch_size
        if epoch in save_epochs or epoch == num_epochs-1: #save last and save periodicly
            now = datetime.now()
            dtString = now.strftime("%d-%m-%Y %Hh%Mm%Ss")
            if FilePath[epoch%MaxCheckPoint] != None:
                os.remove(FilePath[epoch%MaxCheckPoint])
            FilePath[epoch%MaxCheckPoint] = SaveCheckPointsPath + '/Vit2GPT2('+dtString+')Epoch_'+str(epoch+epochCheckpointoffset)+'_N_Loss_'+str(loss_name.data.item())+'_L_Loss_'+str(loss_location.data.item())+'_D_Loss_'+str(loss_date.data.item())+'.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, FilePath[epoch%MaxCheckPoint])
            time.sleep(1)
            scheduler.step()
            # scheduler2.step()
else:
    model.eval()
    n_iters = 0
    model.training = True
    for epoch in range(num_epochs):
        for i, (YName, YLocation, YDate, XFullImage, XCrop, Sizes) in enumerate(train_dataloader if epoch in num_noise_epochs else faketrain_dataloader):
            Images_p = torch.zeros((len(Sizes), max(Sizes) * 3, dataset.encTokens.size, dataset.encTokens.size), dtype=torch.float32)
            for i in range(len(Sizes)):
                Images = dataset.encTokens(images=[XFullImage[i], *XCrop[i][0:Sizes[i] - 1].tolist()], return_tensors="pt")
                Images_p[i][0:3 * Sizes[i]][:][:] = Images["pixel_values"].reshape(1, 3 * Sizes[i], dataset.encTokens.size, dataset.encTokens.size)
            Labels = dataset.decTokens([*YName, *YLocation, *YDate], padding=True, truncation=True, max_length=512, return_tensors="pt")
            output = model(pixel_values=Images_p.to(device),
                           labels=Labels["input_ids"].to(device),
                           decoder_attention_mask=Labels["attention_mask"].to(device),
                           single_loss_sum=True,
                           encoder_batch_lenghs=Sizes)
            if output.loss is not None:
                loss = output.loss
                loss.backward()
                if output.loss_name is not None and output.loss_location is not None and output.loss_date is not None:
                    loss_name = output.loss_name
                    loss_location = output.loss_location
                    loss_date = output.loss_date
                else:
                    loss_name = loss
                    loss_location = loss
                    loss_date = loss
            elif output.loss_name is not None and output.loss_location is not None and output.loss_date is not None:
                loss_name = output.loss_name
                loss_location = output.loss_location
                loss_date = output.loss_date
                loss_date.backward(retain_graph=True)
                loss_location.backward(retain_graph=True)
                loss_name.backward()
            else:
                raise ValueError("No Labels were provided")
            clear()
            text = dataset.decode(output.decoder_outputs.logits_name)
            print(50 * "-" + "Name" + (200 - len("Name")) * "-")
            print("Labels originals:")
            print(YName)
            print("Labels Prediction:")
            print(text)
            print("\n Training loss =" + str(loss_name.data.item()) + "|| epoch =" + str(
                epoch + epochCheckpointoffset) + "/" + str(
                num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                n_iters))
            print(50 * "-" + "end" + (200 - len("end")) * "-")
            text = dataset.decode(output.decoder_outputs.logits_location)
            print(50 * "*" + "Begin_Location" + (200 - len("Begin_Location")) * "*")
            print("Labels originals:")
            print(YLocation)
            print("Labels Prediction:")
            print(text)
            print("\n Training loss =" + str(loss_location.data.item()) + "|| epoch =" + str(
                epoch + epochCheckpointoffset) + "/" + str(
                num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                n_iters))
            print(50 * "*" + "end" + (200 - len("end")) * "*")
            text = dataset.decode(output.decoder_outputs.logits_date)
            print(50 * "+" + "Begin_Date" + (200 - len("Begin_Date")) * "+")
            print("Labels originals:")
            print(YDate)
            print("Labels Prediction:")
            print(text)
            print("\n Training loss =" + str(loss_date.data.item()) + "|| epoch =" + str(
                epoch + epochCheckpointoffset) + "/" + str(
                num_epochs + epochCheckpointoffset) + "|| iteration =" + str(
                n_iters))
            print(50 * "+" + "end" + (200 - len("end")) * "+")
            n_iters += batch_size