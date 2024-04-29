import json
import matplotlib.pyplot as plt

DIRECTORY = 'fliponly/'
epoch_ran = 50

with open(DIRECTORY+'resnetplotting.json', 'r') as fp:
    plotting_data = json.load(fp)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.title('Train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(plotting_data['epoch'], plotting_data['train_loss'])

plt.subplot(1, 2, 2)
plt.title('Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(plotting_data['epoch'], plotting_data['val_loss'])

plt.show()

tail_acc_over_epoch = {}

for i in range(1, epoch_ran+1):
    with open(DIRECTORY+'byclassepoch{}.json'.format(i), 'r') as f:
        epoch_data = json.load(f)
    for threshold in epoch_data:
        total_class = 0
        total_tail = 0
        for each_class in epoch_data[threshold]:
            total_class += 1
            if 'TP' not in epoch_data[threshold][each_class]:
                epoch_data[threshold][each_class]['TP'] = 0
            if 'FP' not in epoch_data[threshold][each_class]:
                epoch_data[threshold][each_class]['FP'] = 0
            if epoch_data[threshold][each_class]['FP'] == 0 and epoch_data[threshold][each_class]['FP'] == 0:
                tail = 0
            else:
                tail = epoch_data[threshold][each_class]['TP']/(
                    epoch_data[threshold][each_class]['TP']+epoch_data[threshold][each_class]['FP'])
            total_tail += tail
        avg_tail = total_tail/total_class
        if threshold not in tail_acc_over_epoch:
            tail_acc_over_epoch[threshold] = [avg_tail]
        else:
            tail_acc_over_epoch[threshold].append(avg_tail)

plt.figure(figsize=(20, 20))
plt.title('Tail Accuracy over epoch for each threshold')
plt.xlabel('epoch')
plt.ylabel('Tail accuracy')
for threshold in tail_acc_over_epoch:
    plt.plot(plotting_data['epoch'], tail_acc_over_epoch[threshold])

plt.legend(tail_acc_over_epoch.keys())
plt.show()
