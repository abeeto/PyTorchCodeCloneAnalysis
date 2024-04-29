import numpy, torch, csv, pandas

myCSV = numpy.loadtxt("winequality-white.csv", dtype=numpy.float16, delimiter=';', skiprows=1)
col_list = next(csv.reader(open("winequality-white.csv"), delimiter=';'))

print(myCSV.shape, col_list)

myTensor = torch.tensor(myCSV).cuda()
print(myTensor.shape, myTensor.type())

source = myTensor[:, :-1]
target = myTensor[:, -1]

targetOneHot = torch.zeros(target.shape[0], 10).cuda()

targetOneHot.scatter_(1, target.unsqueeze(1).long(), 1.0)

import Standardize

myTensorNormal = Standardize.standardize(myTensor)

badIndices = torch.le(target, 3)
badIndices.dtype, badIndices.shape, badIndices.sum()

badData = myTensor[badIndices]
midData = myTensor[torch.gt(target, 3) & torch.le(target, 7)]
goodData = myTensor[torch.gt(target, 7)]

badMean = torch.mean(badData, dim=0)
midMean = torch.mean(midData, dim=0)
goodMean = torch.mean(goodData, dim=0)

for i, args in enumerate(zip(col_list, badMean, midMean, goodMean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

sulfurThreshold = 141.83
sulfurData = myTensor[:, 6]
predictedIndices = torch.lt(sulfurData, sulfurThreshold)

trueIndices = torch.gt(target, 5)

nMatches = torch.sum(trueIndices & predictedIndices).item()
nPredicted = torch.sum(predictedIndices).item()
nTrue = torch.sum(trueIndices).item()

nMatches, nMatches / nPredicted, nMatches / nTrue
