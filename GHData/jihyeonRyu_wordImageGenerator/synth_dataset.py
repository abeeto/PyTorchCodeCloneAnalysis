from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import random
import tqdm

class SynthDataset(Dataset):

    def __init__(self, data_dir, height, width, num_sample=50000, transform=None, debug=False):

        super(SynthDataset, self).__init__()
        self.data_dir = data_dir
        self.img_height = height
        self.img_width = width
        self.num_sample = num_sample
        self.num_font = 1016
        self.min_char = 1
        self.max_char = 25
        self.tmp_dir = os.path.join(self.project_root_path(), "./resource")
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        self.count = 0
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.target_ratio = height / float(width)
        self.paths = []
        self.labels = []
        self.num_dict = {}
        self.char_dict = {}
        self.make_dict()

        self.debug = debug
        self.init_data()

    def project_root_path(self):

        path = os.path.abspath(os.path.dirname(__file__))
        path = "/".join(path.split('/'))

        return path

    def make_dict(self):

        for i in range(1, 63):
            dir_name = "Sample{0:03d}".format(i)
            if i <= 10:
                self.num_dict[dir_name] = str(i-1)
            elif i<=36: # 대문자 (11-36)
                self.char_dict[dir_name] = chr(i+54)
            else:
                self.char_dict[dir_name] = chr(i+60)

    def init_data(self):

        self.labels = []
        self.paths = []
        self.count = 0

        print("Start initialize dataset ... ")
        for i in tqdm.tqdm(range(self.num_sample)):

            num_char = np.random.randint(self.min_char, self.max_char, 1)[0]
            font = np.random.randint(1, self.num_font, 1)[0]
            char_dirs = np.random.choice(list(self.num_dict.keys()), num_char)
            np.random.shuffle(char_dirs)
            char_paths = []
            label = ""
            for i in range(0, num_char):
                filename = "img{}".format(char_dirs[i].split("ple")[1]) + "-{0:05d}.png".format(font)
                full_path = os.path.join(self.data_dir, char_dirs[i], filename)

                if not os.path.exists(full_path):
                    continue
                image = cv2.imread(full_path, 0)
                h, w = image.shape[:2]
                width = int(self.img_height / h * w)
                if width == 0:
                    continue

                char_paths.append(full_path)
                label += self.num_dict[char_dirs[i]]

            if len(label)!=0:
                image = self.make_word_image(char_paths)
                path = os.path.join(self.tmp_dir,"{}.png".format(str(self.count)))
                cv2.imwrite(path, image)
                self.labels.append(label)
                # self.paths.append(char_paths)
                self.paths.append(path)
                self.count+=1

            num_char = np.random.randint(self.min_char, self.max_char, 1)[0]
            font = np.random.randint(1, self.num_font, 1)[0]
            char_dirs = np.random.choice(list(self.char_dict.keys()), num_char)
            np.random.shuffle(char_dirs)
            char_paths = []
            label = ""
            for i in range(0, num_char):
                filename = "img{}".format(char_dirs[i].split("ple")[1]) + "-{0:05d}.png".format(font)
                full_path = os.path.join(self.data_dir, char_dirs[i], filename)

                if not os.path.exists(full_path):
                    continue
                image = cv2.imread(full_path, 0)
                h, w = image.shape[:2]
                width = int(self.img_height / h * w)
                if width == 0:
                    continue

                char_paths.append(full_path)
                label += self.char_dict[char_dirs[i]]

            if len(label) != 0:
                image = self.make_word_image(char_paths)
                path = os.path.join(self.tmp_dir, "{}.png".format(str(self.count)))
                cv2.imwrite(path, image)
                self.labels.append(label)
                # self.paths.append(char_paths)
                self.paths.append(path)
                self.count += 1

    def keepratio_resize(self, img):
        try:
            cur_ratio = img.shape[1] / float(img.shape[0])
            mask_height = self.img_height
            mask_width = self.img_width
            img = np.array(img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if cur_ratio < self.target_ratio:
                cur_target_height = self.img_height
                cur_target_width = self.img_width
            else:
                cur_target_height = self.img_height
                cur_target_width = int(self.img_height * cur_ratio)

            img = cv2.resize(img, (cur_target_width, cur_target_height))
            start_x = int((mask_height - img.shape[0])/2)
            start_y = int((mask_width - img.shape[1])/2)
            mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
            mask[start_x : start_x + img.shape[0], start_y : start_y + img.shape[1]] = img
            img = mask
        except Exception:
            img = cv2.resize(img, (self.img_width, self.img_height))
        return img

    def random_dotted(self, img):

        choice = np.random.choice([0, 1, 2], 1)[0]
        if choice == 0:
            return img
        else:
            step = np.random.randint(int(img.shape[0]/12), int(img.shape[0]/2), size=1)[0]
            index = 0
            if choice == 1:
                while True:
                    index += step
                    if index >= img.shape[0]:
                        break
                    for i in range(img.shape[1]):
                        if img[index, i] == 0: # char region
                            img[index, i] = 255
            else:
                while True:
                    index += step
                    if index >= img.shape[0]:
                        break
                    img[index, :] = 0
        return img

    def __len__(self):
        return len(self.labels)

    def make_word_image(self, char_dirs):

        images = cv2.imread(char_dirs[0], 0)
        h, w = images.shape[:2]
        width = int(self.img_height / h * w)
        images = cv2.resize(images, (width, self.img_height))

        if len(char_dirs) != 1:
            for i in range(1, len(char_dirs)):
                image = cv2.imread(char_dirs[i], 0)
                h, w = image.shape[:2]
                width = int(self.img_height / h * w)
                image = cv2.resize(image, (width, self.img_height))
                images = np.hstack([images, image])

        images = self.random_dotted(images)
        # images = self.random_contrast(images)

        return images

    def random_contrast(self, img):

        # alpha = float(random.uniform(1.0, 2.5)) # gain: contrast
        # beta = int(random.uniform(50, 125))  # bias: brightness
        if random.choice([True, False]):
            return img
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                # img[y, x] = np.clip(alpha * img[y, x] + beta, 0, 255)
                alpha = int(random.uniform(0, 70))
                img[y, x] = np.clip(img[y, x] - alpha, 0, 255)
                beta = int(random.uniform(0, 70))
                img[y, x] = np.clip(img[y, x] + beta, 0, 255)
        return img

    def __getitem__(self, item):
        images = cv2.imread(self.paths[item], 0)
        label = self.labels[item]

        image = self.keepratio_resize(images)

        image = Image.fromarray(image)
        image = self.transform(image)

        if self.debug:
            tmp = (image.data.numpy() * 255).astype(np.uint8).reshape(self.img_height, self.img_width)
            print(label)
            cv2.imshow("test", tmp)
            cv2.waitKey(0)

        sample = {'image': image, "label": label}
        return sample


if __name__ == "__main__":

    dataset = SynthDataset('path to font images', 32, 256, 100, transforms.Compose([transforms.ToTensor()]) , debug=True)
    for d in dataset:
        print("---")
