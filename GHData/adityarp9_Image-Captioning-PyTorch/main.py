"""
    Main training and evaluation code

"""
import time

import dataset as ds
import model as mdl

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vis
import numpy as np
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

from copy import copy, deepcopy


def main():
    dir_photos = "./data/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/"
    file_annot = "./data/Flickr8k/Flickr8k_text/Flickr8k.token.txt"

    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Get basic dataset info
    print("DATASET INFO")
    print("---------------------------------------------------------------------------------------------------------\n")
    jpg_files = ds.images_info(dir_photos)
    print("Number of photos in Flickr8k: %d" % (len(jpg_files)))
    ann_dframe = ds.annots_info(file_annot, df=True)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Visualize data overview
    print("DATASET OVERVIEW")
    print("---------------------------------------------------------------------------------------------------------\n")
    print(ann_dframe)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Prepare captions
    print("CURATE CAPTIONS")
    print("---------------------------------------------------------------------------------------------------------\n")
    word_count = ds.word_freq(ann_dframe)
    # print(word_count)

    ## Clean text
    start = time.time()
    print("Cleaning text ... ", end="")
    for i, cpt in enumerate(ann_dframe.caption.values):
        ann_dframe["caption"].iloc[i] = ds.clean_text(cpt)
    print("done.")
    # print(ann_dframe)
    # word_count = ds.word_freq(ann_dframe)
    # print(word_count)

    ## Add start and end sequence token
    ann_dframe_orig = copy(ann_dframe)
    print("Adding start and end tokens ... ", end="")
    ann_dfrm = ds.add_start_end_tokens(ann_dframe)
    print("done.")
    elapsed = time.time() - start
    print("\nTime to preprocess {} captions: {:.2f} \
            seconds".format(i, elapsed))
    # print(ann_dfrm)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    # ## Read images with specified transforms
    print("READ IMAGES & EXTRACT FEATURES")
    print("---------------------------------------------------------------------------------------------------------\n")
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    transforms = vis.transforms.Compose([vis.transforms.Resize(256), vis.transforms.CenterCrop(224),
                                         vis.transforms.ToTensor(), vis.transforms.Normalize(mean=mean, std=stdv)])
    print("Reading images ... ", end='')
    images = ds.read_image(dir_photos, transforms)
    print("done.")

    # Get feature maps for image tensor through VGG-16
    features_dict, features_fname = mdl.get_features(images, download_wts=False, save=True, cuda=True)
    # print(features_dict)

    ## Load feature maps
    features_dict = torch.load(features_fname)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Prep image tensor
    print("PREP IMAGE TENSOR")
    print("---------------------------------------------------------------------------------------------------------\n")
    ann_dfrm = ann_dfrm.loc[ann_dfrm["idx"].values == "0", :]
    print(ann_dfrm)
    ds.word_freq(ann_dfrm)
    fnames = []
    img_tns_list = []
    cap_list = []
    for i, jpg_name in enumerate(ann_dfrm.filename.values):
        if jpg_name in features_dict.keys():
            fnames.append(jpg_name)
            img_tns_list.append(features_dict[jpg_name])
            cap_list.append(ann_dfrm.iloc[i]["caption"])
    print(len(img_tns_list), len(cap_list))
    img_tns = torch.cat(img_tns_list)
    print(img_tns.shape)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Text tokenize
    print("TEXT TOKENIZE")
    print("---------------------------------------------------------------------------------------------------------\n")
    tokens, cap_seq, vocab_size, cap_max_len = ds.tokenizer(cap_list)
    print("Vocab size: ", vocab_size)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Dataset splits
    print("DATASET SPLIT")
    print("---------------------------------------------------------------------------------------------------------\n")
    n_cap = len(cap_seq)
    vald_prop, test_prop = 0.2, 0.2
    n_vald = int(n_cap * vald_prop)
    n_test = int(n_cap * test_prop)
    train_cap, valid_cap, evaln_cap = ds.split_dset(cap_seq, n_vald, n_test)
    train_ims, valid_ims, evaln_ims = ds.split_dset(img_tns, n_vald, n_test)
    train_fnm, valid_fnm, evaln_fnm = ds.split_dset(fnames, n_vald, n_test)

    print(len(train_cap), len(valid_cap), len(evaln_cap))
    print(len(train_ims), len(valid_ims), len(evaln_ims))
    print(len(train_fnm), len(valid_fnm), len(evaln_fnm))
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Prep data for training and validation
    print("FINAL PREP FOR TRAINING & VALIDATION")
    print("---------------------------------------------------------------------------------------------------------\n")
    images_train, captions_train, target_caps_train = ds.prep_data(train_ims, train_cap, cap_max_len)
    images_valid, captions_valid, target_caps_valid = ds.prep_data(valid_ims, valid_cap, cap_max_len)
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## TRAINING
    print("TRAINING")
    print("---------------------------------------------------------------------------------------------------------\n")

    ## Hyperparameters
    bs = 64
    lr = 0.001
    lr_steps = 20
    gamma = 0.1
    max_n_epochs = 5

    ## Dataloader
    print("DATALOADERS")
    trainset = ds.Flickr8k(images_train, captions_train, target_caps_train)
    validset = ds.Flickr8k(images_valid, captions_valid, target_caps_valid)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bs)

    ## Device: CPU or GPU?
    print("DEVICE:", end=" ")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using " + device)

    ## Model
    print("MODEL:")
    model = mdl.CapNet(vocab_size, cap_max_len).to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    ## Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_steps, gamma=gamma)

    ## Training
    print("\nStarting training ... ")

    epoch_train_loss, epoch_valid_loss = [], []
    min_val_loss = 100
    for epoch in range(1, max_n_epochs + 1):
        print("-------------------- Epoch: [%d / %d] ----------------------" % (epoch, max_n_epochs))
        training_loss, validation_loss = 0.0, 0.0
        ## Batch training
        for i, data in enumerate(trainloader):
            tr_images, tr_captions, tr_target_caps = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            tr_out = model(tr_images, tr_captions.t())
            tr_loss = criterion(tr_out, tr_target_caps)
            tr_loss.backward()
            optimizer.step()
            training_loss += tr_loss.item()
        epoch_train_loss.append(training_loss / len(trainloader))
        print("Training loss: %f" % (epoch_train_loss[-1]), end=" || ")
        for i, data in enumerate(validloader):
            with torch.set_grad_enabled(False):
                vl_images, vl_captions, vl_target_caps = data[0].to(device), data[1].to(device), data[2].to(device)
                vl_out = model(vl_images, vl_captions.t())
                vl_loss = criterion(vl_out, vl_target_caps)
                validation_loss += vl_loss.item()
        epoch_valid_loss.append(validation_loss / len(validloader))
        print("Validation loss: %f" % (epoch_valid_loss[-1]))
        scheduler.step(epoch=epoch)

        if epoch_valid_loss[-1] < min_val_loss:
            print("Found best model.")
            best_model = deepcopy(model)
            min_val_loss = epoch_valid_loss[-1]

    plt.plot(list(range(max_n_epochs)), epoch_train_loss, label="Training loss")
    plt.plot(list(range(max_n_epochs)), epoch_valid_loss, label="Validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Number of epochs vs loss")
    plt.legend()
    plt.show()

    ## Save model
    print("Saving best model ... ")
    torch.save(best_model, 'best_model.pkl')
    print("\n-------------------------------------------------------------------------------------------------------\n")

    ## Check output
    print("Loading model ...")
    model = torch.load('best_model.pkl')
    print(model)

    model.eval()
    preds = []
    for feat in evaln_ims:
        preds.append(model.prediction(feat, tokens, device))

    best_targets = []
    bleu_scores = []
    for p, t in zip(preds, cap_list[:n_test]):
        pred = p.split(" ")
        targ = [t.split(" ")]
        z = sentence_bleu(targ, pred, weights=(1, 0, 0, 0))
        bleu_scores.append(z)
        if z > 0.50:
            print(p, t, z, sep='\n')
            print("\n")
            best_targets.append((p, t, z))
    for i, tgt in enumerate(best_targets):
        print ("{}: {}".format(i, tgt))
    print("MEAN BLEU SCORE: %3f" % np.mean(bleu_scores))

    # for cap in best_targets:
    #     rows = ann_dfrm.loc[ann_dfrm["caption"] == cap, "filename"]
    #     print(rows)


if __name__ == '__main__':
    main()
