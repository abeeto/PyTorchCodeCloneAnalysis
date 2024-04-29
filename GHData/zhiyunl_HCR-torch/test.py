from net_8 import *

"""
================================================================
========================== Function ============================
================================================================
"""


def test_cnn(net, test_data):
    test_data = preprocessor(test_data, 52, 52)
    predict_labels = net8TestOn(net, test_data, device)

    return predict_labels

"""
================================================================
====================== Test and Result==========================
================================================================
"""

if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load true labels
    trainLabel = load_npy("finalLabelsTrain.npy")
    wanyuLabel = load_npy("wanyulabels.npy")
    test_label = trainLabel[6400 - 640:6400]
    EasyLabel = np.concatenate([[1] * 775, [2] * (1565 - 775)])

    # TODO Load Test Data
    EasyData = load_pkl("EasyData.pkl")
    HardData = load_pkl("HardData.pkl")
    trainData = load_pkl("train_data.pkl")
    wanyuData = np.load("wanyudata.npy", allow_pickle=True)
    test_data = trainData[6400 - 640:6400]

    # init net
    net = Net_8().to(device)

    epoch = 600
    # # To find the best epoch model
    # for i in np.arange(epoch):
    #
    #     print('==== Epoch {} ==== '.format(i))
    #
    #     # Load cnn model
    #     netPath = "checkpoint/epoch" + str(i) + ".pth"
    #     net.load_state_dict(torch.load(netPath))
    #
    #     # TODO get prediction
    #     wanyuP = test_cnn(net, wanyuData)
    #     testP = test_cnn(net, test_data)
    #
    #     # Print accuracy
    #     print("wanyuData acc : ", getAcc(wanyuLabel, wanyuP) * 100, "%")
    #     print("testData acc : ", getAcc(test_label, testP) * 100, "%")

    maxNet = 305
    netPath = "checkpoint/epoch" + str(maxNet) + ".pth"
    net.load_state_dict(torch.load(netPath))
    EasyP = test_cnn(net, EasyData)
    HardP = test_cnn(net, HardData)
    print("EasyData acc : ", getAcc(EasyLabel, EasyP) * 100, "%")

    print("End Test")
