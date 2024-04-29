import os
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import shutil
import glob
from pprint import pprint
import random
from pathlib import Path
from tqdm import tqdm

random.seed(0)

mnist_data = MNIST(root='./', train=True, transform=None, download=True)

def makeMnistPng(image_dsets):
    for idx in tqdm(range(10)):
        print("Making image file for index {}".format(idx))
        num_img = 0
        dir_path = './mnist_all/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for image, label in image_dsets:
           if label == idx:
                filename = dir_path +'/mnist_'+ str(idx) + '-' + str(num_img) + '.png'
                if not os.path.exists(filename):
                    image.save(filename)
                num_img += 1
    print('Success to make MNIST PNG image files. index={}'.format(idx))

class FileControler(object):
    def get_file_path(self, input_dir, pattern):
        #ファイルパスの取得
        #ディレクトリを指定しパスオブジェクトを生成
        path_obj = Path(input_dir)
        #glob形式でファイルをマッチ
        files_path = path_obj.glob(pattern)
        #文字列として扱うためposix変換
        files_path_posix = [file_path.as_posix() for file_path in files_path]
        return files_path_posix
    
    def random_sampling(self, files_path, sample_num, output_dir, fix_seed=True) -> None:
        #ランダムサンプリング
        #毎回同じファイルをサンプリングするにはSeedを固定する
        if fix_seed is True:
            random.seed(0)
        #ファイル群のパスとサンプル数を指定
        files_path_sampled = random.sample(files_path, sample_num)
        #出力先ディレクトリがなければ作成
        os.makedirs(output_dir, exist_ok=True)
        #コピー
        for file_path in files_path_sampled:
            shutil.copy(file_path, output_dir)

file_controler =FileControler()

makeMnistPng(mnist_data)

all_file_dir = './mnist_all/'
sampled_dir = './mnist_sampled/'

pattern = '*.png'
files_path = file_controler.get_file_path(all_file_dir, pattern)

sample_num = 100
file_controler.random_sampling(files_path, sample_num, sampled_dir)

files = glob.glob("./mnist_sampled/*")

for i in range(10):
    os.makedirs(sampled_dir+str(i), exist_ok=True)
    for x in files:
        if '_' + str(i) in x:
            shutil.move(x, sampled_dir + str(i))
