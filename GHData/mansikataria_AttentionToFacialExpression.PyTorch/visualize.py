import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import skimage
from skimage.transform import resize
from torch import nn


class Visualize:
    def __init__(self, emotion_labels):
        self.incorrect_examples = []
        self.model_weights = []  # we will save the conv layer weights in this list
        self.conv_layers = []  # we will save the 18 conv layers in this list
        self.emotion_labels = emotion_labels

    def plotTrainValLoss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(val_losses, label="val")
        plt.plot(train_losses, label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plotTrainValAccuracy(self, train_acc, val_acc):
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy")
        plt.plot(val_acc, label="val")
        plt.plot(train_acc, label="train")
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def plotConfusionMatrix(self, model, Testloader):
        model.eval()
        correct = 0
        total = 0
        all_target = []
        for batch_idx, (inputs, targets) in enumerate(Testloader):
            # print('batch_idx: ', batch_idx)
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = model(inputs)

            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            _, predicted = torch.max(outputs_avg.data, 1)

            total += targets.size(0)
            idxs_mask = ((predicted == targets) == False).nonzero()
            # print("idxs_mask: ", idxs_mask)
            # print(idxs_mask.size()[0])
            if (idxs_mask.size()[0] != 0):
                # print(inputs[idxs_mask].cpu().numpy().shape)
                # print(emotion_labels[predicted])
                # print(targets)
                self.incorrect_examples.append(
                    incorrectoutcome(inputs[idxs_mask].cpu().numpy(), self.emotion_labels[targets],
                                     self.emotion_labels[predicted],
                                     batch_idx))

            correct += predicted.eq(targets.data).cpu().sum()
            if batch_idx == 0:
                all_predicted = predicted
                all_targets = targets
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, targets), 0)

        acc = 100. * correct / total
        print("accuracy: %0.3f" % acc)

        # Compute confusion matrix
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        self.plot_confusion_matrix(matrix, classes=self.emotion_labels, normalize=True,
                              title="Private_Test " + ' Confusion Matrix (Accuracy: %0.3f%%)' % acc)
        plt.show()
        # plt.savefig(os.path.join(path, "Private_Test" + '_cm.png'))
        # plt.close()

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.tight_layout()

    def showIncorrectClassifications(self, n=10):
        fig = plt.figure(figsize=(20, 8))

        for idx in np.arange(n):
            ax = fig.add_subplot(2, n / 2, idx + 1, xticks=[], yticks=[])
            # std = np.array([0.229, 0.224, 0.225])
            # mean = np.array([0.485, 0.456, 0.406])
            incorrectoutcome = self.incorrect_examples[idx]
            img = incorrectoutcome.img
            # print(img.shape)
            img = np.transpose(img, (3, 4, 0, 1, 2))
            # print(img.shape)
            img = img[:, :, 0, 0, 0]
            img2 = np.zeros((48, 48, 3))
            img2[:, :, 0] = img
            img2[:, :, 1] = img
            img2[:, :, 2] = img
            plt.imshow(img2)
            ax.set_title(
                f"{incorrectoutcome.predictedLabel}\n (true label: {incorrectoutcome.trueLabel})\n {incorrectoutcome.indexInDataset}",
                color=("green" if incorrectoutcome.predictedLabel == incorrectoutcome.trueLabel else "red"))

    def printAllConvOutputs(self, model, image):
        # counter to keep count of the conv layers
        counter = 0
        model_children = list(model.children())
        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                self.model_weights.append(model_children[i].weight)
                self.conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

        # pass the image through all the layers
        results = [self.conv_layers[0](image)]
        for i in range(1, len(self.conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(self.conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                filter = filter.cpu()
                if i == 64:  # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter)
                plt.axis("off")
            # print(f"Saving layer {num_layer} feature maps...")
            # plt.savefig(f"../outputs/layer_{num_layer}.png")
            plt.show()
            plt.close()

    def showHeatMap(self, image, model, ncrops=10):
        # predicted_labels.append(idx[0])
        # predicted =  train_loader.dataset.classes[idx[0]]

        # print("Target: " + fname + " | Predicted: " +  predicted)

        # GET output of the last conv layer
        # pass the image through all the layers
        results = [self.conv_layers[0](image)]
        for i in range(1, len(self.conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(self.conv_layers[i](results[-1]))

        # make a copy of the `results`
        conv_outputs = results[-1]
        # print('conv_outputs: ', conv_outputs.shape);
        conv_outputs_avg = conv_outputs.mean(0)  # avg over crops
        conv_outputs_avg.cpu().detach().numpy()

        # Extract weight from trained model
        # params = list(saved_model.parameters())
        weight = model.attn2.op.weight
        # weight = saved_model.layer4[0].conv1.weight

        with torch.no_grad():
            outputs = model(image)
            # print('outputs: ', outputs)
            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
            # print('outputs_avg: ', outputs_avg)
            m = nn.Softmax(dim=-1)
            score = m(outputs_avg.data)
            # print('score : ', score)
            _, predicted = torch.max(outputs_avg.data, 0)
            print('predicted: ', self.emotion_labels[predicted.cpu().numpy()])
            print('id: ', predicted)

        CAMs = self.getCAM(conv_outputs_avg, weight.cpu(), predicted)

        # readImg = org_loc+fname+'.png'
        # img = cv2.imread(readImg)
        print(image.shape)
        _, height, width = image.shape

        # img = Testloader1.dataset[index][0]
        img = np.transpose(image, (1, 2, 0))
        # print(img.shape)
        img = img[:, :, 0]
        img2 = np.zeros((48, 48, 3))
        img2[:, :, 0] = img
        img2[:, :, 1] = img
        img2[:, :, 2] = img

        plt.imshow(img2, alpha=0.5, cmap='jet')
        plt.show()
        # plt.imshow(CAMs[0], alpha=0.5, cmap='jet')
        # plt.show()
        plt.imshow(img2, alpha=0.5, cmap='jet')
        plt.imshow(skimage.transform.resize(CAMs[0], (width, height)), alpha=0.5, cmap='jet');
        plt.show()
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.5 + img * 0.5

        # cv2.imwrite("image_1", result)

    def getCAM(self, feature_conv, weight_fc, class_idx):
        nc, h, w = feature_conv.shape
        weight_fc = weight_fc.detach().numpy()
        print('weight_fc[class_idx]: ', weight_fc[class_idx].shape)
        print('feature_conv.reshape((nc, h*w)): ', feature_conv.reshape((nc, h * w)).shape)
        cam = np.matmul(weight_fc[class_idx], feature_conv.reshape((nc, h * w)).cpu().detach().numpy())
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        return [cam_img]

class incorrectoutcome:
    def __init__(self, img, trueLabel, predictedLabel, indexInDataset):
        self.img = img
        self.trueLabel = trueLabel
        self.predictedLabel = predictedLabel
        self.indexInDataset = indexInDataset