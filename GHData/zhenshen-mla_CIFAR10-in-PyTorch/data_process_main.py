from data_process import data_process


if __name__ == '__main__':
    trainloader, testloader = data_process()
    if (len(trainloader) == 5000 and len(testloader) == 1000) or (len(trainloader) == 500 and len(testloader) == 100):
        print('数据封装完成，共有'+str(len(trainloader))+'批次训练数据， '+str(len(testloader))+'批次测试/验证数据。')
        print(1)
    else:
        print('数据封装失败，请查看batch size 大小。')
        print(2)