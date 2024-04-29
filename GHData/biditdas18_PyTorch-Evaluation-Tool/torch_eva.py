import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorboard
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.plotting import output_notebook, output_file, figure, show, ColumnDataSource
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    torch.multiprocessing.freeze_support()
    print('loop')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    ########### USER CAN CHANGE BATCH SIZE AS PER THEIR USE CASES ######################
    BATCH_SIZE = 4
    ########### USER CAN CHANGE BATCH SIZE AS PER THEIR USE CASES ######################

    ########### PLEASE ENTER THE PATH TO YOUR VALIDATION/TEST PATH HERE ######################
    TEST_DATA_PATH = "Dataset1/Validation/"
    ########### PLEASE ENTER THE PATH TO YOUR VALIDATION/TEST PATH HERE ######################

    ########### USER CAN CHANGE TRANSFORM_IMG PARAMETERS AS PER THEIR USE CASES ######################
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    ########### USER CAN CHANGE TRANSFORM_IMG PARAMETERS AS PER THEIR USE CASES ######################

    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


    ################### USE torch.jit TO SAVE YOUR MODEL TO EVALUATE THE MODEL INDEPENDENT OF THE MODEL CLASS#############
    ################### ALTERNATIVELY YOU CAN UNCOMMENT THE BELOW BLOCK OF CODE AND PASTE YOUR OWN MODEL ARCHITECTURE ####
    # class CNN(nn.Module):
    #   def __init__(self, in_channels=3, num_classes=4):
    #         super(CNN, self).__init__()
    #         self.conv1 = nn.Conv2d(
    #             in_channels=3,
    #             out_channels=8,
    #             kernel_size=(3, 3),
    #             stride=(1, 1),
    #             padding=(1, 1),
    #         )
    #         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    #         self.conv2 = nn.Conv2d(
    #             in_channels=8,
    #             out_channels=16,
    #             kernel_size=(3, 3),
    #             stride=(1, 1),
    #             padding=(1, 1),
    #         )
    #         self.fc1 = nn.Linear(16 * 21 * 21, num_classes)

    #   def forward(self, x):
    #         x = F.relu(self.conv1(x))
    #         x = self.pool(x)
    #         x = F.relu(self.conv2(x))
    #         x = self.pool(x)
    #         x = x.reshape(x.shape[0], -1)
    #         x = self.fc1(x)

    #         return x
    ################### USE torch.jit TO SAVE YOUR MODEL TO EVALUATE THE MODEL INDEPENDENT OF THE MODEL CLASS#############
    ################### ALTERNATIVELY YOU CAN UNCOMMENT THE BELOW BLOCK OF CODE AND PASTE YOUR OWN MODEL ARCHITECTURE ####

    ################### PROVIDE THE PATH TO YOUR SAVED MODEL HERE ###############################
    model = torch.jit.load('modelcnntorch2')
    model.to(device)
    ################### PROVIDE THE PATH TO YOUR SAVED MODEL HERE ###############################

    ################### PREDICTING THE ACCURACY OF THE MODEL ###################################
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_data_loader:
        images,labels = images.cuda(),labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy of the model: %.4f %%' % (100 * correct / total))
    ################### PREDICTING THE ACCURACY OF THE MODEL ###################################

    ################### FUNCTION FOR PLOTTING THE CONFUSION MATRIX #############################
    from sklearn.metrics import confusion_matrix, classification_report
    import itertools
    def plot_confusion_matrix(cm, classes,
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

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
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

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        ################### FUNCTION FOR PLOTTING THE CONFUSION MATRIX #############################


    # Graphical analytics
    nb_classes = len(test_data.classes)


    ################### PLOTTING THE CONFUSION MATRIX USING THE FUNCTION DEFINED ABOVE #############
    predict = []
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_data_loader):
            inputs, classes = inputs.cuda(), classes.cuda()
            outputs = model(inputs)
            preds = outputs.cpu().argmax(1)

            predict.append(preds)

            predict1 = torch.cat(predict)

    # Confusion matrix
    cm = confusion_matrix(test_data.targets, predict1)

    plot_confusion_matrix(cm=cm, classes=test_data.classes)
    ################### PLOTTING THE CONFUSION MATRIX USING THE FUNCTION DEFINED ABOVE #############

    ############## PRINTING CONFUSION MATRIX AGAIN USING SNS HEATMAP ####################
    y_pred_list = []
    with torch.no_grad():
        # print(device)
        model.eval()
        for X_batch, _ in test_data_loader:
            X_batch, _ = X_batch.cuda(), _.cuda()
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    # print(y_pred_list)

    y_pred_list = list(itertools.chain.from_iterable(y_pred_list))


    confusion_matrix_df = pd.DataFrame(confusion_matrix(test_data.targets, y_pred_list), index=test_data.classes,
                                       columns=test_data.classes)

    sns.heatmap(confusion_matrix_df, annot=True)
    plt.show()
    ############## PRINTING CONFUSION MATRIX AGAIN USING SNS HEATMAP ####################

    ################ PRINTING CLASSIFICATION REPORT #####################################
    report = classification_report(test_data.targets, y_pred_list)
    print(report)
    ################ PRINTING CLASSIFICATION REPORT #####################################

    # li = []
    #
    # for X_batch, _ in test_data_loader:
    #     X_batch, _ = inputs.to(device), _.to(device)
    #     li.append(X_batch)

    ##################### GENERATING INTERACTIVE CLASSIFICATION REPORT USING BOKEH TABLE ###########
    report = classification_report(test_data.targets, y_pred_list, output_dict=True)
    df = pd.DataFrame(report).transpose()

    Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns]  # bokeh columns
    data_table = DataTable(columns=Columns, source=ColumnDataSource(df))  # bokeh table

    show(data_table)
    ##################### GENERATING INTERACTIVE CLASSIFICATION REPORT USING BOKEH TABLE ###########


    ##################### GENERATING ROC CURVES #####################################################
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(test_data.targets, y_pred_list, pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(test_data.targets))]
    p_fpr, p_tpr, _ = roc_curve(test_data.targets, random_probs, pos_label=1)

    # matplotlib


    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='cnn')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()
    ##################### GENERATING ROC CURVES #####################################################


################################# FOR GENERATING TENSORBOARD ######################################
# GO TO THE FOLDER THAT HAS TENSORBOARD LOGS USING COMMAND PROMP AND RUN THE FOLLOWING COMMAND
# tensorboard --logdir logs
# AFTER USING THE ABOVE COMMAND A HOST WILL BE GENERATED LIKE BELOW
#  http://localhost:6006/
# PASTE THE ABOVE ADDRESS IN YOUR BROWSER AND HIT ENTER TO VIEW THE TENSORBOARD
################################# FOR GENERATING TENSORBOARD ######################################