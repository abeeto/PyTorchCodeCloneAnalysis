import torch
from skimage.metrics import structural_similarity
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

from ESPCN_sample import ESPCN

# Dataset creation（augumentation無し）
def create_dataset():
    print("\n___Creating a dataset...")
    prc = ['/', '-', '\\', '|']
    cnt = 0
    training_data =[]

    for i in range(60):
        d = "./train/"

        # High-resolution image
        img = np.array(Image.open(d+"train_{}_high.tif".format(i)))

        # Low-resolution image
        low_img = np.array(Image.open(d+"train_{}_low.tif".format(i)))

        training_data.append([img,low_img])

        cnt += 1
        print("\rLoading a LR-images and HR-images...{}    ({} / {})".format(prc[cnt%4], cnt, 60), end='')

    print("\rLoading a LR-images and HR-images...Done    ({} / {})".format(cnt, 60), end='')
    print("\n___Successfully completed\n")

    random.shuffle(training_data)   
    lr_imgs = []
    hr_imgs = []

    for hr, lr in training_data:
        lr_imgs.append(lr)
        hr_imgs.append(hr)

    return np.array(lr_imgs), np.array(hr_imgs)

if __name__ == "__main__":
    
    # モデル呼び出し
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESPCN().to(device)
    model.load_state_dict(torch.load("espcn_model_weight.pth"))

    ssim_list = []

    # データセット
    lr_imgs, hr_imgs = create_dataset()

    # 入力用に変換
    lr_imgs = lr_imgs.astype(np.float32) / 255.0
    lr_imgs = lr_imgs.transpose(0, 3, 1, 2) # channel first

    for i, (lr, hr) in enumerate(zip(tqdm(lr_imgs), hr_imgs)):
        
        lr = lr[np.newaxis, :, :, :]
        lr = torch.from_numpy(lr).to(device)
        
        # 推論
        re = model(lr)
        
        # 元の形式に戻す
        re = re.to('cpu').detach().numpy()
        re = re.transpose(0, 2, 3, 1) # channel last
        re = np.reshape(re, (1200, 1500, 3))
        re = re * 255.0
        re = np.clip(re, 0.0, 255.0).astype(np.uint8)
        
        # ssim算出
        ssim = structural_similarity(hr, re, multichannel=True)
        ssim_list.append(ssim)
        
    ssim_list = np.array(ssim_list)
    print(np.mean(ssim_list))
