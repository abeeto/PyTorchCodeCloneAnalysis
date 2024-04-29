
import cv2
import torch
import torch.utils.data
import os
import numpy as np

root = "/media/sean/HDCL-UT1/rendered_chairs"
dirs = []

for d in os.listdir(root):
    if os.path.isdir(os.path.join(root, d)):
        dirs.append(d)

train_chairs = dirs[:1000]
test_chairs = dirs[1000:]
margin = 100
steps = 10
rot_steps = 5

class ChairsDataset(torch.utils.data.Dataset):
    def __init__(self, chairs):
        self.chairs = chairs
        self.cache = {}



        print("loading dataset...")
        for idx in range(len(self.chairs)):
            print(idx, len(self.chairs))
            renders = os.path.join(root, self.chairs[idx], "renders")
            chair_imgs = os.listdir(renders)
            for chair_name in chair_imgs:
                path = os.path.join(renders, chair_name)
                img = cv2.imread(path)[margin:600 - margin, margin:600 - margin,
                      [2, 1, 0]] / 255.0
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
                self.cache[path] = np.moveaxis(img.astype(np.float32), -1, 0)



    def __len__(self):
        return len(self.chairs)

    def __getitem__(self, idx):
        renders = os.path.join(root, self.chairs[idx], "renders")
        chair_imgs = os.listdir(renders)
        seq = []
        np.random.seed(np.fromstring(os.urandom(24), dtype=np.uint32))
        offset = np.random.randint(len(chair_imgs) - 1)

        for i in range(len(chair_imgs)):
            def get_data(i, basey=0, diff=500):
                chair_idx = i % len(chair_imgs)
                chair_name = chair_imgs[chair_idx]

                tokens = chair_name.split("_")
                rotx = int(tokens[2][1:])
                roty = int(tokens[3][1:])

                diffy = (roty - basey + 360) % 360
                if diffy > 180:
                    diffy -= 360

                if abs(diffy) > diff:
                    return None

                path = os.path.join(renders, chair_name)


                if path not in self.cache:
                    print("adding to cache: ", path)
                    img = cv2.imread(path)[margin:600 - margin, margin:600 - margin,
                          [2, 1, 0]] / 255.0
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
                    self.cache[path] = np.moveaxis(img.astype(np.float32), -1, 0)


                mat = self.cache[path]

                return mat, roty, rotx

            mat1, roty1, rotx1 = get_data(offset)

            offset = 0
            re = None
            while re is None:
                offset += np.random.randint(len(chair_imgs) - 1)
                re = get_data(offset, basey=roty1, diff=500)
            mat2, roty2, rotx2 = re

            diffy = (roty2 - roty1 + 360) % 360

            if diffy > 180:
                diffy -= 360

            diffx = rotx2 - rotx1

            seq.append((mat1, [diffy, diffx], mat2))

            if len(seq) >= steps:
                break

        mat1, act, mat2 = zip(*seq)

        return torch.FloatTensor(np.array(mat1)), torch.FloatTensor(np.array(act)), torch.FloatTensor(np.array(mat2))

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        ChairsDataset(train_chairs),
        batch_size=16, shuffle=True,
        num_workers=32, pin_memory=False)

    for i, (mat1, act, mat2) in enumerate(train_loader):
        cv2.imshow("seq", np.hstack([mat.numpy()[0, ...] for mat in mat1]))
        cv2.waitKey(0)
