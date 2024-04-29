from dataloader import A2D2_Dataset
import cv2
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":

    MAKING_TOTAL = False

    if MAKING_TOTAL :

        training_dataset = A2D2_Dataset("training", size=(512, 400))

        total = np.zeros((400, 512, 3))

        range_ = (list(range(len(training_dataset))))
        random.shuffle(range_)

        for id in tqdm(range_[:2000], total=2000):
            img, target = training_dataset.__getitem__(id)
            total += img
            print()
            print("", np.max(total))

            # cv2.imshow("img", img.numpy().transpose(1, 2, 0))
            # cv2.imshow("target", np.argmax(target.numpy(), axis=0)/16)

            # cv2.waitKey(0)
        np.save("total_A2D2.numpy", total)

    else:

        data = np.load("total_A2D2.numpy.npy")
        data = np.array(data / 2000, dtype=np.uint8)
        data = np.moveaxis(data, -1, 0)

        print(data.mean(axis=(1, 2)))
        print(data.std(axis=(1, 2)))

        # Result
        # [122.15811035 123.63384277 125.46741699]
        # [26.7605721  35.98626225 39.93803676]

        import matplotlib.pyplot as plt
        plt.imshow(data)
        plt.show()
