import json
import os
import re
import sys
from tkinter import filedialog
from typing import List
from datasets import load_metric
import torch
import argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from skimage import io, transform
from YOLOv4.models import main as YOLOv4main
from custom_modeling_Dual_encoder_Multiple_decoder import generate_adaptor
from transformers import ViTFeatureExtractor, GPT2Tokenizer, AutoConfig


def Crop(Square, Imagem, X, Y, Width, Height, BW, White=True):
    if Square:
        if not White:
            if Height % 2 == 0:
                Height += 1

            if Width % 2 == 0:
                Width += 1

            DIFF = abs((Height - Width) / 2)
            DIFF = [DIFF, DIFF]
            if DIFF[0] - int(DIFF[0]) > 0:
                DIFF[0] = int(DIFF[0]) + 1
                DIFF[1] = int(DIFF[1])
            else:
                DIFF[0] = int(DIFF[0])
                DIFF[1] = int(DIFF[1])

            if Height > Width:
                if X - DIFF[0] < 0:
                    DIFF[1] = DIFF[1] + (DIFF[0] - X)
                    DIFF[0] = X

                if (X + Width) + DIFF[1] > Imagem.shape[1]:
                    DIFF[0] = DIFF[0] + (X + Width + DIFF[1] - Imagem.shape[1])
                    DIFF[1] = DIFF[1] - (X + Width + DIFF[1] - Imagem.shape[1])

                if Y < 0:
                    Y = 0

                if Y + Height > Imagem.shape[0]:
                    Height = Imagem.shape[0] - Y

                return Imagem[Y:(Y + Height), (X - DIFF[0]):(X + Width + DIFF[1])]
            else:
                if Y - DIFF[0] < 0:
                    DIFF[1] = DIFF[1] + (DIFF[0] - Y)
                    DIFF[0] = Y

                if (Y + Height) + DIFF[1] > Imagem.shape[0]:
                    DIFF[0] = DIFF[0] + (Y + Height + DIFF[1] - Imagem.shape[0])
                    DIFF[1] = DIFF[1] - (Y + Height + DIFF[1] - Imagem.shape[0])

                if X < 0:
                    X = 0

                if X + Width > Imagem.shape[1]:
                    Width = Imagem.shape[1] - X

                return Imagem[(Y - DIFF[0]):(Y + Height + DIFF[1]), X:(X + Width)]
        else:
            if Height % 2 == 0:
                Height += 1

            if Width % 2 == 0:
                Width += 1

            if Height > Width:
                if BW:
                    ImgBlank = np.zeros([Height, Height, 1], dtype=np.uint8)
                else:
                    ImgBlank = np.zeros([Height, Height, 3], dtype=np.uint8)
            else:
                if BW:
                    ImgBlank = np.zeros([Width, Width, 1], dtype=np.uint8)
                else:
                    ImgBlank = np.zeros([Width, Width, 3], dtype=np.uint8)

            ImgBlank.fill(255)

            ImgBlank[int((ImgBlank.shape[1] - Height) / 2):int((ImgBlank.shape[1] - Height) / 2) + Height,
            int((ImgBlank.shape[0] - Width) / 2):int((ImgBlank.shape[0] - Width) / 2) + Width] = Imagem[Y:(Y + Height), X:(X + Width)]
            return ImgBlank
    else:
        return Imagem[Y:(Y + Height), X:(X + Width)]

# Rescale images for Yolov4
Rescale = True
# Resolution target for Yolov4 (needs to be multiple of 32x32)For less distorsion keep the H/W near 1.414 A4 sheet
# Optional inference sizes:
#   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
#   Width in {320, 416, 512, 608, ... 320 + 96 * m}
Width = 608
Height = 608
Ratio = Height / Width
# Generates a GreyScale(L) DataSet for Yolov4 instead of RGB
BW_Yolo = True
# This sets a threshold that the image will no longe be streched and will add white bars to target the Ratio set above
LowThreshold = 0.87
TopThreshold = 1.13
# Sets the croping for Image Captioning to a square image(Necessary for Tiled ouput)
Square = True
# Generates a GreyScale(L) DataSet for image captioning instead of RGB
BW = False
PrintImage = False
# Set the Filler to white instead of extra parts of the image
White = False
# Sets with you want to run test with all the checkpoints in the folder instead of picking one
RunMultiple = False
# saves a image resulting from YOLOv4 inference:
Draw_YOLOv4 = False

def main(image_path_or_image_dir, YOLOWeightFilePath, TransformerCheckpointFilePath, image_to_label_file_path=None, YOLOClassFilePath=None):
    if os.path.isdir(TransformerCheckpointFilePath):
        if not RunMultiple:
            ListingCheckpoint = [filedialog.askopenfilename(initialdir=TransformerCheckpointFilePath, title="Select Transformer CheckPoint file", filetypes=[('Checkpoint', '*.pth')])]
        else:
            ListingCheckpoint = os.listdir(TransformerCheckpointFilePath)
            ListingCheckpoint: List[str] = list(filter(lambda elem: '.pth' in elem, ListingCheckpoint))
            ListingCheckpoint = [TransformerCheckpointFilePath + "/" + i for i in ListingCheckpoint]
    else:
        ListingCheckpoint = [TransformerCheckpointFilePath]

    if image_to_label_file_path is None:
        if os.path.isdir(image_path_or_image_dir):
            path = image_path_or_image_dir
        else:
            path = os.path.split(image_path_or_image_dir)[0]
        Listing = os.listdir(path)
        Listing: List[str] = list(filter(lambda elem: '.json' in elem, Listing))
        if len(Listing) > 0:
            try:
                image_to_label_file_path = filedialog.askopenfilename(initialdir=path, title="Select a Dataset file", filetypes=[('datasetfile', '*.json')])
            except:
                image_to_label_file_path = None

    if image_to_label_file_path is not None:
        sacrebleu_name = load_metric("sacrebleu")
        sacrebleu_location = load_metric("sacrebleu")
        sacrebleu_date = load_metric("sacrebleu")
        wer_name = load_metric("wer")
        wer_location = load_metric("wer")
        wer_date = load_metric("wer")
        with open(image_to_label_file_path) as dataSetPath_crop:
            data = json.load(dataSetPath_crop)
            Label = {'Name/Family': data["YData"]['Name/Family'],
                'Location': data["YData"]['Location'],
                'Date': data["YData"]['Date']}
            ImagePath = data["XPath"]['FullImage']
            ImageName = [os.path.split(ImPath)[1] for ImPath in ImagePath]

    if os.path.isdir(image_path_or_image_dir):
        ListingImage = os.listdir(image_path_or_image_dir)
        ListingImage: List[str] = list(filter(lambda elem: '.jpeg' in elem or '.png' in elem or '.gif' in elem or '.tiff' in elem or '.raw' in elem or '.jpg' in elem, ListingImage))
        ImagesPath = [image_path_or_image_dir + "/" + i for i in ListingImage]
        try:
            os.mkdir(image_path_or_image_dir + '/YOLOv4')
            Draw_YOLOv4_Results = Draw_YOLOv4
        except:
            print('Folder creation error or already exists ')
            Draw_YOLOv4_Results = False
    else:
        ImagesPath = [image_path_or_image_dir]
        tail, head = os.path.split(image_path_or_image_dir)
        try:
            os.mkdir(tail + '/YOLOv4')
            Draw_YOLOv4_Results = Draw_YOLOv4
        except:
            print('Folder creation error or already exists ')
            Draw_YOLOv4_Results = False

    for CheckpointPath in ListingCheckpoint:
        TransformationINFO = {"Image_Path": str,
                          "leftfill": int, "rightfill": int, "topfill": int, "botfill": int,
                          "transform_width": int, "transform_height": int,
                          "original_width": int, "original_height": int,
                          "Deformation_width": int, "Deformation_height": int}
        model = generate_adaptor(CheckpointPath=CheckpointPath)
        Yolo_Miss = []
        for ImagePath in ImagesPath:
            if image_to_label_file_path is not None:
                head, tail = os.path.split(ImagePath)
                try:
                    index = ImageName.index(tail)
                except:
                    index = None
            Im = io.imread(ImagePath)
            LowThresholdVal = Ratio * LowThreshold
            TopThresholdVal = Ratio * TopThreshold
            TransformationINFO['original_height'] = Im.shape[0]
            TransformationINFO['original_width'] = Im.shape[1]
            if LowThresholdVal < Im.shape[0] / Im.shape[1] < TopThresholdVal:
                TransformationINFO['leftfill'] = 0
                TransformationINFO['rightfill'] = 0
                TransformationINFO['topfill'] = 0
                TransformationINFO['botfill'] = 0
                TransformationINFO['transform_height'] = TransformationINFO['original_height']
                TransformationINFO['transform_width'] = TransformationINFO['original_width']
                Im = Image.fromarray(Im)
            else:
                if Im.shape[0] / Im.shape[1] > TopThresholdVal:
                    tofill = ((Im.shape[0] / Ratio) - Im.shape[1]) / 2
                    if tofill - int(tofill) > 0:
                        TransformationINFO['leftfill'] = (int(tofill)) + 1
                        TransformationINFO['rightfill'] = int(tofill)
                        TransformationINFO['topfill'] = 0
                        TransformationINFO['botfill'] = 0
                    else:
                        TransformationINFO['leftfill'] = int(tofill)
                        TransformationINFO['rightfill'] = int(tofill)
                        TransformationINFO['topfill'] = 0
                        TransformationINFO['botfill'] = 0
                    ImageBlank = np.zeros([Im.shape[0], int(Im.shape[0] / Ratio), 3], dtype=np.uint8)
                    ImageBlank.fill(255)
                    ImageBlank[:, TransformationINFO['leftfill']: TransformationINFO['leftfill'] + Im.shape[1]] = Im[:]
                    Im = Image.fromarray(ImageBlank)
                    TransformationINFO['transform_height'] = ImageBlank.shape[0]
                    TransformationINFO['transform_width'] = ImageBlank.shape[1]
                elif Im.shape[0] / Im.shape[1] < LowThresholdVal:
                    tofill = ((Im.shape[1] * Ratio) - Im.shape[0]) / 2
                    if tofill - int(tofill) > 0:
                        TransformationINFO['leftfill'] = 0
                        TransformationINFO['rightfill'] = 0
                        TransformationINFO['topfill'] = (int(tofill)) + 1
                        TransformationINFO['botfill'] = int(tofill)
                    else:
                        TransformationINFO['leftfill'] = 0
                        TransformationINFO['rightfill'] = 0
                        TransformationINFO['topfill'] = int(tofill)
                        TransformationINFO['botfill'] = int(tofill)
                    ImageBlank = np.zeros([int(Im.shape[1] * Ratio), Im.shape[1], 3], dtype=np.uint8)
                    ImageBlank.fill(255)
                    ImageBlank[TransformationINFO['topfill']: TransformationINFO['topfill'] + Im.shape[0], :] = Im[:]
                    Im = Image.fromarray(ImageBlank)
                    TransformationINFO['transform_height'] = ImageBlank.shape[0]
                    TransformationINFO['transform_width'] = ImageBlank.shape[1]
                else:
                    print('Failed to format image')
                    exit(-1)
            TransformationINFO['Deformation_height'] = TransformationINFO['transform_height'] / Height
            TransformationINFO['Deformation_width'] = TransformationINFO['transform_width'] / Width
            boxes = YOLOv4main(1, YOLOWeightFilePath, np.array(Im.convert('L').convert('RGB')), Height, Width, YOLOClassFilePath)
            Boxes = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int}
            BoxesArray = []
            for box in boxes:
                for point in box:
                    if point[0] < point[2]:
                        Boxes['X1'] = int((point[0] * Width) * TransformationINFO['Deformation_width']) - TransformationINFO['leftfill']
                        Boxes['X2'] = int((point[2] * Width) * TransformationINFO['Deformation_width']) - TransformationINFO['leftfill']
                    else:
                        Boxes['X1'] = int((point[2] * Width) * TransformationINFO['Deformation_width']) - TransformationINFO['leftfill']
                        Boxes['X2'] = int((point[0] * Width) * TransformationINFO['Deformation_width']) - TransformationINFO['leftfill']
                    if point[1] < point[3]:
                        Boxes['Y1'] = int((point[1] * Height) * TransformationINFO['Deformation_height']) - TransformationINFO['topfill']
                        Boxes['Y2'] = int((point[3] * Height) * TransformationINFO['Deformation_height']) - TransformationINFO['topfill']
                    else:
                        Boxes['Y1'] = int((point[3] * Height) * TransformationINFO['Deformation_height']) - TransformationINFO['topfill']
                        Boxes['Y2'] = int((point[1] * Height) * TransformationINFO['Deformation_height']) - TransformationINFO['topfill']
                    BoxesArray.append(Boxes.copy())
            if Draw_YOLOv4_Results:
                YOLOIM = Image.open(ImagePath)
                draw = ImageDraw.Draw(YOLOIM)
            ImageCropGrouped = []
            for Box in BoxesArray:
                ImageCrop = Crop(Square, io.imread(ImagePath), Box['X1'], Box['Y1'], int(abs(Boxes['X2']-Boxes['X1'])), int(abs(Boxes['X2']-Boxes['X1'])), BW, White)
                if Draw_YOLOv4_Results:
                    draw.rectangle(((Box['X1'], Box['Y1']), (Box['X2'], Box['Y2'])), outline="red", width=10)
                if len(ImageCrop) > 0:
                    ImageCropGrouped.append(ImageCrop.copy())
            if Draw_YOLOv4_Results:
                tail, head = os.path.split(ImagePath)
                YOLOIM.save(tail + '/YOLOv4/' + head, "PNG")
            if PrintImage:
                for ImageCrop in ImageCropGrouped:
                    Image.fromarray(ImageCrop).show()
                input()
            try:
                if len(ImageCropGrouped) >= 1:
                    name_output, location_output, date_output = model.generate(Full_Image=Image.open(ImagePath), Crops=ImageCropGrouped)
                    text = name_output
                    print(50 * "-" + "Name" + (200 - len("Name")) * "-")
                    print("Labels Prediction:")
                    print(text[0].lower())
                    if index is not None:
                        try:
                            Name = Label['Name/Family'][index][0] + ", " + Label['Name/Family'][index][1]
                        except:
                            Name = Label['Name/Family'][index][0]
                        sacrebleu_name.add_batch(predictions=[text[0].lower()], references=[[Name.lower()]])
                        wer_name.add_batch(predictions=[text[0].lower()], references=[Name.lower()])
                        print("Labels originals:")
                        print(Name)
                    print(50 * "-" + "end" + (200 - len("end")) * "-")
                    text = location_output
                    print(50 * "-" + "Location" + (200 - len("Location")) * "-")
                    print("Labels Prediction:")
                    print(text[0].lower())
                    if index is not None:
                        try:
                            Location = Label['Location'][index][0] + ", " + Label['Location'][index][1]
                        except:
                            Location = Label['Location'][index][0]
                        sacrebleu_location.add_batch(predictions=[text[0].lower()], references=[[Location.lower()]])
                        wer_location.add_batch(predictions=[text[0].lower()], references=[Location.lower()])
                        print("Labels originals:")
                        print(Location)
                    print(50 * "-" + "end" + (200 - len("end")) * "-")
                    text = date_output
                    print(50 * "-" + "Location" + (200 - len("Location")) * "-")
                    print("Labels Prediction:")
                    print(text[0].lower())
                    if index is not None:
                        Date = Label['Date'][index][0]
                        sacrebleu_date.add_batch(predictions=[text[0].lower()], references=[[Date.lower()]])
                        wer_date.add_batch(predictions=[text[0].lower()], references=[Date.lower()])
                        print("Labels originals:")
                        print(Date)
                    print(50 * "-" + "end" + (200 - len("end")) * "-")
                else:
                    Yolo_Miss.append(ImagePath)
            except:
                print("This model implementation is experimental !!!!! Something Went Wrong during Inference!!!!!!!")
        lines = "Labels Name Score:\n"
        lines += str(sacrebleu_name.compute()).replace(".", ",")
        lines += "\n"
        lines += str(wer_name.compute()).replace(".", ",")
        lines += "\n"
        lines += "Labels Location Score:\n"
        lines += str(sacrebleu_location.compute()).replace(".", ",")
        lines += "\n"
        lines += str(wer_location.compute()).replace(".", ",")
        lines += "\n"
        lines += "Labels Date Score:\n"
        lines += str(sacrebleu_date.compute()).replace(".", ",")
        lines += "\n"
        lines += str(wer_date.compute()).replace(".", ",")
        lines += "\n"
        lines += "Yolo_Miss\n"
        lines += str(len(Yolo_Miss))
        print(lines)
        tail, head = os.path.split(CheckpointPath)
        TextFile = open(tail + '/' + head[:-4] + '.txt', "w")
        TextFile.write(lines)
        TextFile.close()

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='-im : \t path to image or images folder \n -label : \t A optional path to a .json that matches image name to the expected labels for metric calculation \n -yolo : \t A required path to a yolo checkpoint file \n -transformer : \t A required path to a custom Vision encoder decoder model checkpoint file \n -class : \t A optional class file for yolo when its not passed the yolo model will not display output')
    parser.add_argument('-im', type=str, help='A required path to image or images folder to do inference')
    parser.add_argument('-label', type=str, help='A optional path to a .json that contains: [\"YData\"][\'Name/Family\'],[\"YData\"][\'Location\'],[\"YData\"][\'Date\'] and [\"XPath\"][\'FullImage\'] for metric calculation')
    parser.add_argument('-yolo', type=str, help='A required path to a yolo checkpoint file')
    parser.add_argument('-transformer', type=str, help='A required path to a custom Vision encoder decoder model checkpoint file')
    parser.add_argument('-class', type=str, help='A optional class file for yolo when its not passed the yolo model will not display output')
    args = parser.parse_args()
    main(args.im, args.yolo, args.transformer, args.label, args.class)
    '''
    image_path_or_image_dir = 'E:/Users/Halo_/Desktop/thesis/dataset/Dataset_latest_not_not/TestingData'
    image_to_label_file_path = "E:/Users/Halo_/Desktop/thesis/dataset/Dataset_latest_not_not/TestingData/Dataset.json"
    YOLOWeightFilePath = 'E:/Users/Halo_/PycharmProjects/Thesis_test/YOLOv4/checkpoints_work/works/Yolov4_epoch300.pth'
    TransformerCheckpointFilePath = 'E:/Users/Halo_/PycharmProjects/Thesis_test/Google-vit-base-patch16-224-in21ktoGpt2_08-06-2022/Vit2GPT2(09-06-2022 23h08m51s)Epoch_129_N_Loss_6.0299269534880295e-05_L_Loss_1.8504475519875996e-05_D_Loss_1.2077674682586803e-06.pth'
    main(image_path_or_image_dir=image_path_or_image_dir, YOLOWeightFilePath=YOLOWeightFilePath, TransformerCheckpointFilePath=TransformerCheckpointFilePath, image_to_label_file_path=image_to_label_file_path)