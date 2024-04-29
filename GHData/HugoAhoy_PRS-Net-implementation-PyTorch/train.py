import torch
import Network
import MyDataset as MD
from torch.utils.data import DataLoader
import torch.optim as optim
import LossFunction as loss

batchsize = 4
epoch = 100
RegularWeight = 25
LR = 0.01
resolution = 32

def train(epoch, batchsize, RegularWeight, LR):
    prsnet = Network.PRSNet()
    # 设置gpu
    device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    prsnet.to(device)

    optimizer = optim.Adam(prsnet.parameters(), lr = LR)
    LossSym = loss.SymmetryDistanceLoss()
    LossReg = loss.RegularizationLoss()

    dataset = MD.MyDataset("./data")
    dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = True)
    bestLoss = float("inf")
    for e in range(1,epoch):
        totalLoss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            output = prsnet(data['voxel'])

            loss = LossSym(output, data) + RegularWeight*LossReg(output)
            
            totalLoss = totalLoss + float(loss)
            
            loss.backward()
            optimizer.step()
        print("epoch {}, loss:{}".format(e, totalLoss))
        if totalLoss < bestLoss:
            torch.save(prsnet, 'net_best.pkl')

        if e % 10 == 0:
            torch.save(prsnet, 'net{}.pkl'.format(int(e/10)))
    
            
if __name__ == "__main__":
    train(epoch, batchsize, RegularWeight, LR)