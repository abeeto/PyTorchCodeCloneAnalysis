import os
import torch
import math
from torch.utils.data import Dataset, DataLoader

import util


class BatchDatasetSimple(Dataset):
    """Does everything the BatchDataset can do, but is entirely synchronous; 
at the edge of a batch, it will wait for a disk read. Unlike BatchDataset, which 
asynchronously loads two batches at a time, so that you usually do not have to 
wait for a disk read."""

    def __init__(self, formattedDirName, transform = None):
        super(BatchDatasetSimple, self).__init__()
        
        if not util.isDirectoryFormatted(formattedDirName):
            raise OSError("Dir '" + formattedDirName + "' is not formatted for Batch Datasets.\n" + \
                          "Make a formatted directory first using createFormattedDir or download one online.")

        self.dirname = formattedDirName
        self.transform = transform
        self.N, self.batchSize, self.Nb = util.parseManifest(formattedDirName)

        # The last batch might need to be treated specially.
        remainder = self.N % self.batchSize
        if remainder > 0:
            self.batchSize_last = remainder
        else:
            self.batchSize_last = self.batchSize

        self.warmup()


    def warmup(self, batchIndex = 0):
        """Puts this batch in RAM, for faster access."""
        batchIndex = util.handleIndex(batchIndex, self.Nb) # Checks limits and handles negatives.

        self.data = torch.load(os.path.join(self.dirname, 'batch' + str(batchIndex)))
        self.warmBatch = batchIndex


    def __len__(self):
        return self.N

    
    def __getitem__(self, idx):
        idx = util.handleIndex(idx, self.N)
 
        batchInd = idx // self.batchSize

        if batchInd != self.warmBatch: 
            self.warmup(batchInd) # Get it from disk

        dataInd = idx % self.batchSize
        sample = self.data[dataInd]
        if self.transform:
            sample = self.transform(sample)

        return sample

