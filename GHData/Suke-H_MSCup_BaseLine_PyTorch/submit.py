import torch
from PIL import Image
import numpy as np
import zipfile

from ESPCN_sample import ESPCN

if __name__ == "__main__":

    # モデル呼び出し
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESPCN().to(device)
    model.load_state_dict(torch.load("espcn_model_weight.pth"))

    # 推論
    for i in range(40):
        d = "./evaluation/"

        # Low-resolution image
        img = np.array(Image.open(d+"test_{}_low.tif".format(i)))
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, :, :, :]
        img = img.transpose(0, 3, 1, 2) # channel first
        img = torch.from_numpy(img).to(device)

        re = model(img)
        re = re.to('cpu').detach().numpy()
        re = re.transpose(0, 2, 3, 1) # channel last
        re = np.reshape(re, (1200, 1500, 3))
        
        re = re * 255.0
        re = np.clip(re, 0.0, 255.0)
        sr_img = Image.fromarray(np.uint8(re))

        sr_img.save("./output/test_{}_answer.tif".format(i))
        print("Saved ./output/test_{}_answer.tif".format(i))
        
    # zipに圧縮
    with zipfile.ZipFile('submit.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for i in range(40):
            new_zip.write("./output/test_{}_answer.tif".format(i), arcname="test_{}_answer.tif".format(i))
