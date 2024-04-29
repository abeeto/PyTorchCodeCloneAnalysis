from dataProcess import LoadData
from modelStructure import VGG
import torch
folder = "../../Dataset/LeafData1/train"
input_shape = (224,224)

def main():
    torch.cuda.empty_cache()
    """Load data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vgg19 = VGG(classes=3).to(device)
    dataLoader = LoadData.LoadDir(folder,size=input_shape,batch_size=8,shuffle=True)
    vgg19.batch_train(dataLoader,epochs=20)
if __name__ == '__main__':
    main()

