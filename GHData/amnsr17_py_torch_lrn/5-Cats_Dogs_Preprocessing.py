import os
import cv2
import numpy as np
from tqdm import tqdm

# to do preprocessing only once we are making a flag
REBUILD_DATA = True

class DogsVsCats():
    # desired image size
    IMG_SIZE = 50
    # data paths
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS:0, DOGS:1}
    # labeled data will be populated in this list
    training_data = []
    # counters
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        """ Accesses each image in Cat and Dog categories,
         gray scale it, resizes it and add it to the training_data along with its label"""
        for label in self.LABELS:
            print(label)
            for file in tqdm(os.listdir(label)):
                # some images will refuse to load. for this try-except is being used.
                if "jpg" in file:
                    try:
                        # making the full path of the image by joining
                        image_path = os.path.join(label, file)
                        # reading the image and gray scaling it
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        # resizing
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # adding this image to the training data along with its class
                        # class labels are in one-hot vector format
                        # np.eye(total_num_of_classes)[index_to_make_hot]
                        # self.LABELS[label] gives the value of the label which is 0 for cat and 1 for dog
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                        # print(np.eye(2)[self.LABELS[label]])

                        # counter increment
                        if label==self.CATS:
                            self.catcount += 1
                        elif label==self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        # Shuffling data after loading and labeling
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:",self.catcount)
        print("Dogs:", self.dogcount)


if REBUILD_DATA:
    dogvcats = DogsVsCats()
    dogvcats.make_training_data()


# Loading the saved training data
training_data = np.load("training_data.npy", allow_pickle=True)
print("Total Training Examples:",len(training_data))

# checking
import matplotlib.pyplot as plt
plt.imshow(training_data[16000][0], cmap="gray")
plt.show()

print(training_data[16000][1])
# Next: Batching, and passing through CNN


