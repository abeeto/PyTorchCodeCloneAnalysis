
import torch


class Callback_reduceLearningRateOnPlateau():
    def __init__(self, patience=5, coef=0.5, cooldown=5):
        self.patience = patience
        self.coef = coef
        self.cooldown = cooldown
        
        self.lowestLoss = None
        self.lowestLossEpochNumber = None
        self.lastLearningRateDecrease = None

    def __call__(self, learningRate, loss, epochNumber):
        if (self.lowestLoss is None):
            self.lowestLoss = loss
            self.lowestLossEpochNumber = epochNumber

        if (loss < self.lowestLoss):
            self.lowestLoss = loss
            self.lowestLossEpochNumber = epochNumber
        
        updateLearningRate = False
        if (epochNumber - self.lowestLossEpochNumber >= self.patience):
            if (self.lastLearningRateDecrease is None):
                updateLearningRate = True
            elif (epochNumber - self.lastLearningRateDecrease >= self.cooldown):
                updateLearningRate = True

        if (updateLearningRate):
            self.lastLearningRateDecrease = epochNumber
            learningRate = learningRate * self.coef
            print(f"\n[Callback_reduceLearningRateOnPlateau] New LR = {learningRate}")

        return learningRate



class Callback_learningRateWarmup():
    def __init__(self, warmupLength=5, coef=0.1):
        self.warmupLength = warmupLength
        self.coef = coef
    def __call__(self, learningRate, epochNumber):
        newLearningRate = False
        if (epochNumber == 0):
            for _ in range(self.warmupLength):
                learningRate *= self.coef
                newLearningRate = True
        elif (epochNumber <= self.warmupLength):
            learningRate /= self.coef
            newLearningRate = True
        if (newLearningRate):
            print(f"[Callback_learningRateWarmup] New LR = {learningRate}")
        return learningRate



class Callback_computeValidationMetrics():
    def __init__(self, validationDataloader, criterion, device):
        self.validationDataloader = validationDataloader
        self.device = device
        self.criterion = criterion

    def __call__(self, model):
        model.eval() # Deactivate Batchnorm & Dropout layers
        with torch.no_grad():

            averagedValidationLoss = 0

            for batchNumber, (xBatch, yBatch) in enumerate(self.validationDataloader):

                xBatch = torch.from_numpy(xBatch.astype('float32'))
                yBatch = torch.from_numpy(yBatch.astype('float32'))

                xBatch = xBatch.to(self.device)
                yBatch = yBatch.to(self.device)

                yBatch[:, :, 8] = 0  # Remove trajectory

                outputs = model(xBatch)

                # Loss
                loss = self.criterion(outputs, yBatch)
                averagedValidationLoss += loss.item()
        
            averagedValidationLoss /= (batchNumber+1)

        return averagedValidationLoss



class Callback_unfreezeNetwork():
    def __init__(self, model, initialDelay=5, unfreezePeriod=3, numberOfUnfreezingSteps=10):
        self.model = model
        self.initialDelay = initialDelay
        self.unfreezePeriod = unfreezePeriod
        self.numberOfUnfreezingSteps = numberOfUnfreezingSteps

        self.unfreezeEpochNumber = [(initialDelay + unfreezePeriod * stepNumber) for stepNumber in range(self.numberOfUnfreezingSteps)]
        self.unfreezeAmounts = [(stepNumber+1) / numberOfUnfreezingSteps for stepNumber in range(numberOfUnfreezingSteps)]

    def __call__(self, epochNumber):
        try :
            stepIndex = self.unfreezeEpochNumber.index(epochNumber)
        except:
            return

        unfreezeAmount = self.unfreezeAmounts[stepIndex]
        params = [x for x in self.model.parameters()]
        for param in params[int(unfreezeAmount * len(params)):]:
            param.requires_grad = True
        print(f"[Callback_unfreezeNetwork] Last {int(unfreezeAmount * len(params))} layers unfrozen")



