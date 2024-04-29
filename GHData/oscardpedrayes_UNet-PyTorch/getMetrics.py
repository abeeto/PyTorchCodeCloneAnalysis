import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import sys 
import os

if len(sys.argv) < 3:
    print("Wrong usage. getMetrics.py Dir Epochs")
    exit()


dir_ = sys.argv[1] + '/'
epochs = int(sys.argv[2])
print('\n',dir_)


trainloss= []
testloss= []
testga= []


for fil in os.listdir(dir_):
    if "tfevents" in fil:
        # Iterate throught tags, get global metrics first
        for e in summary_iterator(dir_ + fil):
            #print(e)
            for v in e.summary.value:               
                if 'Loss/train'  in v.tag :
                    trainloss.append(v.simple_value)
                elif 'Loss/test' in v.tag:
                    testloss.append(v.simple_value)
                elif 'GlobalAccuracy/test' in v.tag:
                    testga.append(v.simple_value)

                

trainloss = trainloss[::len(trainloss)//epochs]
plt.plot(trainloss)
plt.plot(testloss)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.ylim(0.0,1.5)
plt.grid(True)
plt.savefig(dir_+'/loss.svg')
plt.close()

plt.plot(testga)
plt.title('GlobalAccuracy en Test')
plt.ylabel('globalaccuracy')
plt.xlabel('epoch')
plt.ylim(0.0,1)
plt.grid(True)
plt.savefig(dir_+'/ga.svg')
plt.close()

