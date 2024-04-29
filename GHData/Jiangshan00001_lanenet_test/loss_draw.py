import json
import matplotlib.pyplot as plt
if __name__=='__main__':
    f=open('lanenet_loss.txt','r')
    loss_list = json.loads(f.read())
    f.close()
    plt.plot(loss_list)
    plt.xlabel('iter/10')
    plt.ylabel('loss')
    plt.title('lanenet loss')
    plt.savefig('./pics/lanenet_loss.png')
    plt.show()

