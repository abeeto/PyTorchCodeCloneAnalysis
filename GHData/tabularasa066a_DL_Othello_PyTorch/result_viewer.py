"""
    validation_loss.csvを基にValidation Loss-Epochsグラフを描画
    第一列がepoch、第二列が損失関数の値
"""

import csv
import matplotlib.pyplot as plt

tr_epochs_list = []
train_loss_list = []
with open('train_loss.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        epoch = float(row[0])
        train_loss = float(row[1])
        tr_epochs_list.append(epoch)
        train_loss_list.append(train_loss)

val_epochs_list = []
validation_loss_list = []
validation_acc_list = []
with open('validation_loss.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        epoch = float(row[0])
        validation_loss = float(row[1])
        validation_acc = float(row[2])
        val_epochs_list.append(epoch)
        validation_loss_list.append(validation_loss)
        validation_acc_list.append(validation_acc)

## validation accuracyの最大値をパーセント換算で表示
best_acc = max(validation_acc_list)
best_acc_percent = best_acc * 100
idx_when_best_acc = validation_acc_list.index(best_acc)
print("Best accuracy: ", best_acc_percent, "%, on the epoch: ", int(val_epochs_list[idx_when_best_acc]))

## Train時の損失関数値
plt.title("Epochs-Train Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Value of Train Loss Function")
# plt.xlim(0,len(tr_epochs_list))
plt.plot(tr_epochs_list, train_loss_list)
plt.show()
plt.close()

## Validation時の損失関数値
plt.title("Epochs-Validation Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Value of Validation Loss Function")
# plt.xlim(0,len(tr_epochs_list))
plt.plot(val_epochs_list, validation_loss_list)
plt.show()
plt.close()

## Validation時の正答率
plt.title("Epochs-Validation Accuracy Plot")
plt.xlabel("Epochs")
plt.ylabel("Value of Validation Accuracy Function")
plt.plot(val_epochs_list, validation_acc_list)
plt.show()
plt.close()
