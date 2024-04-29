import os
from PIL import Image
import chainer
import shutil
import glob
from pprint import pprint
import random
from pathlib import Path

random.seed(0)

def save(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    filename = "./mnist_all/" + str(num) + "/" + str(num) + "_test" + "{0:05d}".format(index) + ".png"
    img.save(filename)
    print(filename)

def main():
    train, _ = chainer.datasets.get_mnist()
    for i in range(10):
        dirpath = "./mnist_all/" + str(i)
        if os.path.isdir(dirpath) is False:
            os.makedirs(dirpath)
    for i in range(len(train)):
        save(train[i][0], i, train[i][1])

if __name__ == '__main__':
    main()

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

for i in range(10):
    all_file_dir = './mnist_all/' + str(i)
    sampled_dir = './mnist_sampled/' + str(i)

    pattern = '*.png'
    files_path = file_controler.get_file_path(all_file_dir, pattern)
    pprint(files_path)

    sample_num = 60
    file_controler.random_sampling(files_path, sample_num, sampled_dir)