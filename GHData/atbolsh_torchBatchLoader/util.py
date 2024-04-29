import os
import torch
import math

def handleIndex(ind, limit):
    if ind < 0:
        ind = limit + ind
    if ind < 0 or ind > limit - 1:
        raise IndexError("Index out of bounds")
    return ind

def writeManifest(dirname, totalRecords, batchSize, numBatches):
    """Writes the correctly formatted manifest."""
    f = open(os.path.join(dirname, 'manifest'), 'w')
    s = "This directory is formatted for a BatchDataset.\n"
    s += "This file contains metadata.\n\n"
    s += "The total number of records is " + str(totalRecords) + "\n"
    s += "The batchsize is " + str(batchSize) + "\n"
    s += "The total number of batch fiies written is " + str(numBatches) + "\n"
    f.write(s)
    f.close()
    return True


def parseManifest(dirname):
    """Parses the manifest file into totalRecords, batchSize, and numBatches. 
If there is no manifest file, will throw an error; bad formatting will lead to 
undefined behavior."""
    try:
        f = open(os.path.join(dirname, 'manifest'), 'r')
    except FileNotFoundError:
        raise OSError("Directory '" + dirname + "' has no manifest, is not a formatted directory.")
    s = f.read()
    f.close()
    l = s.split('\n')[3:]
    totalRecords = int(l[0].split(' ')[-1])
    batchSize = int(l[1].split(' ')[-1])
    numBatches = int(l[2].split(' ')[-1])
    return totalRecords, batchSize, numBatches


def isDirectoryFormatted(dirname):
    """Checks that only the manifest and the correct number of batch files are present. 
However, it does not verify the contents of the batchs; storing incorrect batches leads 
to undefined behavior."""
    try:
        _, _, numBatches = parseManifest(dirname)
    except OSError:
        return False

    l = os.listdir(dirname)
    l.remove('manifest') # Only batches should remain

    # Fastest check: check length
    if len(l) != numBatches:
        return False

    # Check that each name begins with 'batch'
    for fname in l:
        if fname[:5] != 'batch':
            return False
    
    # Check that all the suffixes are correct
    inds = [int(fname[5:]) for fname in l]
    inds.sort()
    for i in range(numBatches):
        if inds[i] != i:
            return False

    # We passed all checks.
    return True


def verifyDirEmpty(dirname, cleanDir=False):
    """Checks if directory is empty. 
If not, and cleanDir is set to True, empties the directory 
and still returns True."""
    if len(os.listdir(dirname)) > 0:
        if cleanDir:
            for fname in os.listdir(dirname):
                os.remove(os.path.join(dirname, fname))
        else:
            return False
    return True 


def createFormattedDir(source, dirname, batchSize = 128, cleanDir=False, verbose=True):
    """Given a source that supports __len__ and __getitem__, 
and a directory that's empty, create a formatted directory that's 
then used by BatchDataset. This only needs to be run once, so it's a utility. 
Adjust batchSize so two batch leaves enough (CPU) RAM for normal processes, 
but is as large as possible. BathDataset sometimes loads two batches, 
to avoid waiting on a disk read."""
    if verifyDirEmpty(dirname, cleanDir=cleanDir):
        if verbose:
            print("Directory '" + dirname + "' is empty.")
    else:
        raise OSError("Directory '" + dirname  + "' not empty; choose a new directory or set cleanDir=True.")

    totalRecords = len(source)
    numBatches = math.ceil(totalRecords / batchSize)
    if writeManifest(dirname, totalRecords, batchSize, numBatches) and verbose:
        print("Manifest written.")

    def _writeBatch(container, recordId):
        batchId = recordId // batchSize
        pathName = os.path.join(dirname, 'batch' + str(batchId))
        torch.save(container, pathName)
        if verbose:
            print("Records up to " + str(recordId + 1) +  " saved; " + str(batchId + 1) + " batches written.")

    # Main loop to store all the records.
    # By default, python list are the batches, but the BatchDataset class will work with any indexable container.
    res = []
    for i in range(totalRecords):
        res.append(source[i])
        if (i+1) % batchSize == 0:
            _writeBatch(res, i)
            res = []

    if len(res) > 0: # Take care of stragglers.
        _writeBatch(res, i)
        res = []
   
    print("Finished! Directory '" + dirname + "' is ready to be used for a BatchDataset.")
    return True 

