import numpy as np 
import torch 
import torch.optim as optim
import os
import json
import random

from time import time
import callbacks



def main():

    # Training parameters
    optimizerNames = ["adam"]
    learningRates = [1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4]
    epoques = [20, 50]
    batchSizes = [1024, 2048, 4096]
    weightDecays = [0, 1e-6]
    noiseAmplitudes = [0, 0, 0, 0, 1e-6]
    pads = [2, 5, 10]
    learningRateDecays = [0.9]#, 0.8]

    # Network parameters
    hiddenLayerSize = [1024, 2048, 4096]
    numbersOfConvolutions = [0, 1, 2, 5]
    numbersOfFCs = [1]
    resnetStyles = [True]
    activations = [None]#, "relu"] # Relu is still to be added

    while True :

        parameters = {}

        parameters["optimizerName"] = random.choice(optimizerNames)
        parameters["learningRate"] = random.choice(learningRates)
        parameters["epoque"] = random.choice(epoques)
        parameters["batchSize"] = random.choice(batchSizes)
        parameters["weightDecay"] = random.choice(weightDecays)
        parameters["noiseAmplitude"] = random.choice(noiseAmplitudes)
        parameters["pad"] = random.choice(pads)
        parameters["learningRateDecay"] = random.choice(learningRateDecays)
        parameters["hiddenLayerSize"] = random.choice(hiddenLayerSize)
        parameters["numberOfConvolutions"] = random.choice(numbersOfConvolutions)
        parameters["numberOfFCs"] = random.choice(numbersOfFCs)
        parameters["resnetStyle"] = random.choice(resnetStyles)
        parameters["activation"] = random.choice(activations)

        # Manual testing
        """
        parameters["optimizerName"] = "adam"
        parameters["learningRate"] = 1e-3
        parameters["epoque"] = 20
        parameters["batchSize"] = 1024
        parameters["weightDecay"] = 1e-6
        parameters["noiseAmplitude"] = 1e-5
        parameters["pad"] = 10
        parameters["learningRateDecay"] = 0.9
        parameters["hiddenLayerSize"] = 1024
        parameters["numberOfConvolutions"] = 1
        parameters["numberOfFCs"] = 1
        parameters["resnetStyle"] = True
        parameters["activation"] = None
        """

        # Fixed parameters
        parameters["causal"] = True
        parameters["dropout"] = 0.75
        parameters["densnetStyle"] = False

        print("====================================== New training ======================================")

        print("parameters : ", parameters)

        train(parameters)

def train(parameters):


# =================================================================================================================================================
# ============================================================== Step 1 : Parameters ==============================================================
# =================================================================================================================================================

    os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device 0 as seen by cuda

    #--------------Train parameters-----------------
    # DA : flip data or not
    # Stride : chunk length, should be one for training
    stride = 1
    data_augmentation = True
    epochNumber = 0

    savingFolderPath = f'/temporary/Saved_Trainings/{parameters["optimizerName"]}_nconv{parameters["numberOfConvolutions"]}_nfc{parameters["numberOfFCs"]}_res{parameters["resnetStyle"]}_act{parameters["activation"]}_lr{parameters["learningRate"]}_lrDecay{parameters["learningRateDecay"]}_bs{parameters["batchSize"]}_wd{parameters["weightDecay"]}_noise{parameters["noiseAmplitude"]}_pad{parameters["pad"]}_hsize{parameters["hiddenLayerSize"]}_{random.random()}/'
    directory = os.path.dirname(savingFolderPath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# ============================================================================================================================================
# ============================================================== Step 2 : Model ==============================================================
# ============================================================================================================================================

    model = myModel_GridSearch(
        numberOfConvolutions=parameters["numberOfConvolutions"], 
        numberOfFCs=parameters["numberOfFCs"], 
        hiddenLayerSize=parameters["hiddenLayerSize"], 
        activation=parameters["activation"], 
        pad=parameters["pad"], 
        resnetStyle=parameters["resnetStyle"], 
        denseNetStyle=False
    )
    model_params = 0
    for parameter in model2D3D.parameters():
        model_params += parameter.numel()

    model.to(device)
    
    # ==============================================================================================================================================
    # ============================================================== Step 3 : Dataset ==============================================================
    # ==============================================================================================================================================

    train_generator = None
    valid_generator = None
    test_generator = None

    # ================================================================================================================================================
    # ============================================================== Step 4 : Optimizer & Loss & Metrics =============================================
    # ================================================================================================================================================

    # if (optimizerName == "adam"): optimizer = optim.Adam(model2D3D.parameters(), lr=lr, weight_decay=1e-5) #, amsgrad=True)
    if (parameters["optimizerName"] == "adam"): optimizer = optim.Adam(model2D3D.parameters(), lr=parameters["learningRate"], weight_decay=parameters["weightDecay"]) #, amsgrad=True)
    #if (parameters["optimizerName"] == "adam"): optimizer = optim.Adam(model2D3D.parameters(), lr=parameters["learningRate"]) #, amsgrad=True)
    if (parameters["optimizerName"] == "sgd"): optimizer = torch.optim.SGD(model2D3D.parameters(), lr=parameters["learningRate"], momentum=0.9, weight_decay=parameters["weightDecay"])
    if (parameters["optimizerName"] == "adadelta"): optimizer = torch.optim.Adadelta(model2D3D.parameters(), lr=parameters["learningRate"], weight_decay=parameters["weightDecay"])

    # Scheduler should be phrased as a callback ?
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, parameters["learningRateDecay"])

    criterion = None

    # Error measure
    trainLosses = []
    validLosses = []

    # ================================================================================================================================================
    # ============================================================== Step 5 : Callbacks ==============================================================
    # ================================================================================================================================================

    callback_computeValidationMetrics = callbacks.Callback_computeValidationMetrics(test_generator, mpjpe, device)
    callback_learningRateWarmup = callbacks.Callback_learningRateWarmup(warmupLength = 3, coef = 0.5)
    callback_reduceLearningRateOnPlateau = callbacks.Callback_reduceLearningRateOnPlateau(patience=3, coef=0.5, cooldown=3)

    # ===============================================================================================================================================
    # ============================================================== Step 6 : Training ==============================================================
    # ===============================================================================================================================================

    # Start training
    print("Start training")
    while epochNumber < parameters["epoque"]:
        epochNumber += 1

        print(f"\nEpoch {epochNumber}")

        time_epochStart = time()
        epochLoss = 0
        numberOfFrames_epoch = 0
        model.train()

        for batchNumber, (xBatch, yBatch) in enumerate(train_generator):
            print(f"\r\tBatch {batchNumber}", end='')

                # Prepare data
            noise = np.random.normal(0, parameters["noiseAmplitude"],(xBatch.shape))
            xBatch = xBatch + noise
            xBatch = torch.from_numpy(xBatch.astype('float32'))
            xBatch = xBatch.to(device)
            yBatch = torch.from_numpy(yBatch.astype('float32'))
            yBatch = yBatch.to(device)

                # Optimize weigths
            optimizer.zero_grad()
            predictions = model(xBatch)
            batchLoss = criterion(predictions, yBatch)
            batchLoss.backward()
            optimizer.step()

            parameters["learningRate"] *= parameters["learningRateDecay"] # lr decay as a callback ?
        scheduler.step()
        print()

        # Record epoch-wise average loss
        trainLosses.append(None)

        # Compute validation loss
        averagedValidationLoss = callback_computeValidationMetrics(model2D3D)
        # print("averagedValidationLoss : ", averagedValidationLoss)
        validLosses.append(averagedValidationLoss)

        print("trainLosses : ", trainLosses)    
        print("validLosses : ", validLosses)

    # ===============================================================================================================================================
    # ============================================================== Step 7 : Testing ===============================================================
    # ===============================================================================================================================================

    # TODO

    # ===============================================================================================================================================
    # =============================================================== Step 8 : Saving ===============================================================
    # ===============================================================================================================================================


        if (epochNumber in [5, 10, 20, 50]):
            losses = {"train" : trainLosses, "valid" : validLosses}
            with open(f"{savingFolderPath}losses.json", "w") as saveFile:
                saveFile.write(json.dumps(losses))

            with open(f"{savingFolderPath}parameters.json", "w") as saveFile:
                saveFile.write(json.dumps(parameters))

            # Save checkpoint
            savePath = f"{savingFolderPath}model_ep{epochNumber}.pt"
            torch.save({
                'epoch': epochNumber,
                'lr': parameters["learningRate"],
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model2D3D.state_dict()
            }, savePath)



if (__name__ == "__main__"):
    main()
