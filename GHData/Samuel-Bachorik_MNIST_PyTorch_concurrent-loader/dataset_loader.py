import os, os.path
from PIL import Image
import numpy
import random
import torch
import concurrent.futures
import math
class ImagesLoader:
    def __init__(self, batch):
        self.batch = batch

    def loop(self,paths):

        labels0 = numpy.zeros(self.batch, dtype=numpy.int64)
        imgs = numpy.zeros((self.batch, 1, 28, 28), dtype=numpy.float32)

        for i in range(self.batch):

            rand_path = random.randint(0, 9)
            path = paths[rand_path]
            labels0[i] = rand_path

            f = os.listdir(path)
            x = Image.open(os.path.join(path, f[random.randint(0, len(f) - 1)]))

            img = numpy.asarray(x)

            img = numpy.expand_dims(img, axis=0)
            g = (img / 255.0).astype(numpy.float32)

            g = g.squeeze(0)

            imgs[i] = g.copy()

        return labels0, imgs

    def get_dataset(self,paths,training):
        print("Loading dataset...")
        if training == True:
            epoch = 60000/self.batch
            print("Loading training dataset")
        else:
            epoch = 10000/self.batch
            print("Loading testing dataset")

        imgs2 = numpy.zeros((math.ceil(epoch), self.batch, 1, 28, 28), dtype=numpy.float32)
        labels = numpy.zeros((math.ceil(epoch), self.batch), dtype=numpy.int64)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [None] * math.ceil(epoch)
            for x in range(math.ceil(epoch)):
                results[x] = executor.submit(self.loop,paths)


            counter = 0
            for f in concurrent.futures.as_completed(results):
                imgs2[counter], labels[counter] = f.result()[1], f.result()[0]
                counter += 1


        return torch.from_numpy(imgs2), torch.from_numpy(labels)
