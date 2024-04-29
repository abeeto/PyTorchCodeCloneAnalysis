from torch.utils.data import Dataset
import os
import Sendr

class DataLoader(Dataset):
    """The DataSet is assumed to be in a hierarchical form.
    DataSet Folder -> (Test and Train)
    Test -> Different Label Folder -> Training images
    Train -> Different Label Folders -> Training images
    For a text data give the .csv location"""
    def __init__(self, DFolder, transforms = None, istext = False):
        self.DFolder = DFolder
        self.transforms = transforms
        self.istext = istext
        labels = walk(DFolder = self.DFolder, labels = True)
        features = walk(DFolder = self.DFolder, features = True)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pass

    def walk(self, DFolder, labels = False, features = False):
        """This function returns a list of labels or features in a
        Dataset. This gets the data in a linear form from a treeish form"""
        Data_Dict = Dict()
        walked = list(os.walk(DFolder))
        for file_ in walked:
            if os.path.isdir(file_[0]):
                Data_Dict.update({})
            else:
                pass
        if labels == True:
            return Data_Dict.keys()
        if features == True:
            f = []
            for keys, values in Data_Dict.items():
                f.append([values[0], values[1]])
            return f
        else:
            return
