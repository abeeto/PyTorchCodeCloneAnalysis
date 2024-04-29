import torchvision
import torchvision.datasets

import util as u

originalDataDir = "CelebA/celeba/img_align_celeba/"
formattedDir = "sampleFormattedDir"
#originalDataDir = "CelebA_mini/"
#formattedDir = "sampleFormattedDir_mini"

originalDataset = torchvision.datasets.ImageFolder(root = originalDataDir)

u.createFormattedDir(originalDataset, formattedDir, batchSize = 128, cleanDir = True)


