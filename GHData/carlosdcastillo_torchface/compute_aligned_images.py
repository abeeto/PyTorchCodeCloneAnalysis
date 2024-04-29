# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy
from numpy import matrix
from numpy import linalg
from collections import defaultdict
from multiprocessing import Pool as ThreadPool
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm


def load_csv(filename):
    data = []
    titles = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            d = {}
            if i == 0:
                for item in row:
                    titles.append(item)
            else:
                for j, item in enumerate(row):
                    d[titles[j]] = item

                data.append(d)
        # print titles
    return (data, titles)


def find_coeffs_similarity(pa, pb):

    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0])
        matrix.append([p1[1], -p1[0], 0, 1])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(2 * len(pa))

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    res = res[0][0]
    T = numpy.matrix(
        [[res[0, 0], res[0, 1], res[0, 2]],
            [-res[0, 1], res[0, 0], res[0, 3]]]
    )

    return (T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2])



def process(item):

    basedir = '/fs/janus-scratch/carlos/umdfaces/umdfaces_batch3/'
    filename = os.path.join(basedir, item['FILE'])

    outputdir = '/scratch2/umdfaces-thumbnails/'
    end = '/'.join(filename.split('/')[-2:])

    dirp = '/'.join(end.split('/')[:-1])
    if random.random() < 0.1:
        folder = os.path.join(outputdir, 'val', dirp)
        fno = os.path.join(outputdir, 'val', dirp,  filename.split('/')[-1])
    else:
        folder = os.path.join(outputdir, 'train', dirp)
        fno = os.path.join(outputdir, 'train', dirp,  filename.split('/')[-1])

    if not os.path.exists(fno):

        try:
            im = Image.open(filename).convert('RGB')
        except:
            return

        li = []
        for i in range(1, 22):
            li.append((float(item['P%dX' % i]), float(item['P%dY' % i])))

        idx = [6, 8, 9, 11, 14, 17, 19]
        q = []
        for x in idx:
            q.append(li[x])

        #-----
        ref = []
        ref.append((50.0694, 68.3160))
        ref.append((68.3604, 68.3318))
        ref.append((88.3886, 68.1872))
        ref.append((106.9256, 67.6126))
        ref.append((78.7128,86.0086))
        ref.append((62.2908, 106.0550))
        ref.append((95.7594, 105.5998))

        coeffs = find_coeffs_similarity(q, ref)

        x0 = -0.8762 - 0.25
        y0 = -7.3191 - 0.25

        T = matrix([[coeffs[0], coeffs[1], coeffs[2] - x0],
                    [coeffs[3], coeffs[4], coeffs[5] - y0], [0, 0, 1]])
        Tinv = linalg.inv(T)
        Tinvtuple = (Tinv[0, 0], Tinv[0, 1], Tinv[0, 2],
                     Tinv[1, 0], Tinv[1, 1], Tinv[1, 2])

        im = im.transform((200, 200), Image.AFFINE,
                          Tinvtuple, resample=Image.BILINEAR)
        im = im.crop((8, 8, 152, 152))
        im = im.resize((256, 256), resample=Image.BILINEAR)

        #-----

        if not os.path.isdir(folder):
            try:
                os.makedirs(folder)
            except:
                print('probably: dir exists', folder)

        try:
            im.save(fno)
        except:
            return


def main():
    print('loading')
    (data, titles) = load_csv(
        '/fs/janus-scratch/ankan/umdfaces_images/umdfaces_batch3/umdfaces_batch3_ultraface.csv')
    listofitems = []
    for item in data:
        listofitems.append(item)
    print('loaded')
    total = len(listofitems)
    pool = ThreadPool(processes=16, maxtasksperchild=1000)
    for item in tqdm(pool.imap_unordered(process, listofitems), total=total):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
