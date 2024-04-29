from readers import StandfordDataSetReader, GloveReader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net 
import numpy as np

def train(data_info, model_arch, training_params, num_epochs, log_interval):
    glove = GloveReader(path =  data_info['GLOVE'])
    embeddings, word2id = glove.get()


    sd = StandfordDataSetReader(path = data_info['SD'])
    data, labels = sd.get("train", word2id)
    data = torch.utils.data.DataLoader(data, batch_size = training_params['BATCH_SIZE'])

    model = Net(model_arch, vocab_size = len(word2id), embedding_dim = 300, use_embeddings = training_params["USE_EMBEDDINGS"], embeddings = embeddings)
    optimizer = optim.SGD(model.parameters(), lr = training_params["LR"], momentum = training_params["MM"], weight_decay = training_params["WD"])

    training_loss = []
    for epoch in range(1, num_epochs + 1):
        print("Epoch: " + str(epoch))
        
        model.train()
        loss_list = []
        acc_list = []

        for i, batch in enumerate(data):
            if len(labels) < i*training_params['BATCH_SIZE'] + training_params['BATCH_SIZE']:
                break

            batch_labels = labels[i*training_params['BATCH_SIZE'] : training_params['BATCH_SIZE']*(i + 1)]
            batch, batch_labels = batch.to(training_params['DEVICE']), batch_labels.to(training_params['DEVICE'])
            batch_labels = batch_labels.view((training_params['BATCH_SIZE'],))
            optimizer.zero_grad()
            output = model(batch.long())
            loss = F.cross_entropy(output, batch_labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(batch_labels.view_as(pred)).float().mean()
            acc_list.append(acc.item())

            if i % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}".format(
                        epoch, i * training_params['BATCH_SIZE'], len(data.dataset),
                               100*i*training_params['BATCH_SIZE'] / len(data.dataset), np.mean(loss_list), np.mean(acc_list)))

                training_loss.append(np.mean(loss_list))
                loss_list.clear()
                acc_list.clear()
                

    torch.save(model.state_dict(), "model.pt")

    return training_loss
