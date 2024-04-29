
import numpy as np 
import torch 
#import torch.optim as optim
import torch.nn as nn



def main():

    pad = 5
    model = LSTM_SeqToOne_GridSearch(numberOfConvolutions=3, numberOfFCs=5, hiddenLayerSize=1024, activation="sigmoid", pad=pad, resnetStyle=True, denseNetStyle=False)

    fakeInput = torch.randn(1024, 2*pad+1, 15, 2)  
    fakePred = model(fakeInput, verbose=True)
    print("fakePred.shape : ", fakePred.shape)



class LSTM_SeqToOne_GridSearch(nn.Module):
    def __init__(self, numberOfConvolutions=0, numberOfFCs=1, hiddenLayerSize=1024, activation=None, pad=2, resnetStyle=False, denseNetStyle=False, dropoutRate=0.75, inputSize=30, outputSize=45):
        super(LSTM_SeqToOne_GridSearch, self).__init__()
        self.sequenceLenght = 2 * pad + 1
        self.hiddenLayerSize = hiddenLayerSize
        self.num_joints_out = outputSize//3
        self.activation = activation
        self.resnetStyle = resnetStyle
        
        convolutionLayerSize = hiddenLayerSize

        # ======================== Convolutions ========================
        self.convolutions = []
        self.batchNorms = []
        for convolutionIndex in range(numberOfConvolutions):

            if (convolutionIndex == 0): # First convolution must handle input
                convolution = torch.nn.Conv1d(inputSize, convolutionLayerSize, kernel_size=3, stride=1, padding=1)
            else:
                convolution = torch.nn.Conv1d(convolutionLayerSize, convolutionLayerSize, kernel_size=3, stride=1, padding=1)

            self.convolutions.append(convolution)
            self.batchNorms.append(torch.nn.BatchNorm1d(convolutionLayerSize))

            # Make lists visible by pytorch (for selecting the parameters, for example)
        self.convolutions = nn.ModuleList(self.convolutions)
        self.batchNorms = nn.ModuleList(self.batchNorms)

        # ============================ LSTM ============================
        if (numberOfConvolutions == 0): # LSTM must handle input if no convolution before
            lstmLayer = nn.LSTM(inputSize, hiddenLayerSize)
        else:
            lstmLayer = nn.LSTM(convolutionLayerSize, hiddenLayerSize)
        self.lstm = lstmLayer
        self.lstm_dropout = nn.Dropout(dropoutRate)
        self.lstm_sigmoid = torch.nn.Sigmoid()

        # ============================= FCs =============================
        self.fcs = []
        self.dropouts = []
        self.activations = []
        for fcIndex in range(numberOfFCs):

            # Compute size of FC input
            if (fcIndex == 0): fcInputSize = self.sequenceLenght * hiddenLayerSize # First FC must handle LSTM
            else:
                if (resnetStyle): fcInputSize = (fcIndex * hiddenLayerSize)
                else: fcInputSize = hiddenLayerSize

            # Compute size of output
            if (fcIndex == (numberOfFCs - 1)): fcOutputSize = outputSize # Last FC must predict 3D joints
            else: fcOutputSize = hiddenLayerSize

            self.fcs.append(nn.Linear(fcInputSize, fcOutputSize))
            self.dropouts.append(nn.Dropout(dropoutRate))
            if (activation == "sigmoid"): self.activations.append(torch.nn.Sigmoid())
            if (activation == "relu"): self.activations.append(torch.nn.Relu())
            if (activation is None): self.activations.append(None)

            # Make lists visible by pytorch
        self.fcs = nn.ModuleList(self.fcs)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.activations = nn.ModuleList(self.activations)


    def forward(self, input, verbose=False):
        assert len(input.shape) == 4 # Input has shape [batchSize, SequenceLenght, 15, 2]
        assert input.shape[1] == self.sequenceLenght

            # Fuse 2D coord into a single vector
        x = input.view(input.shape[0],input.shape[1], -1) # [batchSize, SequenceLenght, 15, 2] -> [batchSize, SequenceLenght, 30]
        if (verbose): print("input.shape : ", x.shape)

            # Convolutions
        x = x.permute(0, 2, 1) # [batchSize, SequenceLenght, 30] -> [batchSize, 30, SequenceLenght]
        for convolutionIndex, (convolutionLayer, batchNormLayer) in enumerate(zip(self.convolutions, self.batchNorms)):
            x = convolutionLayer(x)
            x = batchNormLayer(x)
            if (verbose): print(f"conv{convolutionIndex}.shape : ", x.shape)

            # LSTM
        x = x.permute(2, 0, 1) # [batchSize, hiddenLayerSize, SequenceLenght] -> [SequenceLenght, batchSize, hiddenLayerSize]
        x, _ = self.lstm(x) # [SequenceLenght, batchSize, hiddenLayerSize] -> [SequenceLenght, batchSize, hiddenLayerSize]
        x = self.lstm_dropout(x)
        if (verbose): print("lstm.shape : ", x.shape)

            # Flatten (SequenceLenght & hiddenLayerSize) into a single vector
        x = x.permute(1, 0, 2) # [SequenceLenght, batchSize, hiddenLayerSize] -> [batchSize, SequenceLenght, hiddenLayerSize]
        x = x.reshape(x.shape[0], self.sequenceLenght * self.hiddenLayerSize) # [batchSize, SequenceLenght, hiddenLayerSize] -> [batchSize, SequenceLenght * hiddenLayerSize]

            # Compute FCs
        for fcIndex, (fcLayer, dropout, activation) in enumerate(zip(self.fcs, self.dropouts, self.activations)):

            if (self.resnetStyle):
                x_residual = x

            x = fcLayer(x)
            if (fcIndex < (len(self.fcs) - 1)): x = dropout(x)
            if ((activation is not None) and (fcIndex < (len(self.fcs) - 1))): x = activation(x)

            if (self.resnetStyle and (fcIndex != 0) and (fcIndex < (len(self.fcs) - 1))):
                x = torch.cat((x_residual, x), dim=-1) # Also try with simple addition

            if (verbose): print(f"fc{fcIndex}.shape : ", x.shape)

        if (verbose): print("3D_Pose.shape :  ", x.shape)

        output = x.view(x.shape[0],1,self.num_joints_out,3)
        return output



if __name__ == '__main__':
    main()
