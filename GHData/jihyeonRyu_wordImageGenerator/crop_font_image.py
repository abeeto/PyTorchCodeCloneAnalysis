import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='', type=str)
parser.add_argument('--target_path', default='', type=str)
args = parser.parse_args()

def crop_font_image(dir_path, target_path):
    
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    dirs = os.listdir(dir_path)
    margin = 5
    for d in dirs:

        if not os.path.exists(os.path.join(target_path, d)):
            os.mkdir(os.path.join(target_path, d))

        files = os.listdir(os.path.join(dir_path, d))

        for file in files:
            
            full_path = os.path.join(dir_path, d, file)
            try:
                image = cv2.imread(full_path)
                tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, tmp = cv2.threshold(tmp, 10, 255, cv2.THRESH_BINARY)
                t, l, b, r = 0, 0, tmp.shape[0]-1, tmp.shape[1]-1
                for i in range(0, tmp.shape[0]):
                    if 0 in tmp[i, :]:
                        t = i-margin
                        break
                for i in range(tmp.shape[0]-1, -1, -1):
                    if 0 in tmp[i, :]:
                        b = i+margin
                        break
                for i in range(0, tmp.shape[1]):
                    if 0 in tmp[:, i]:
                        l = i - margin
                        break
                for i in range(tmp.shape[1]-1, -1, -1):
                    if 0 in tmp[:, i]:
                        r = i+margin
                        break
                image = image[t:b, l:r, :]
                # cv2.imshow("test",image)
                # cv2.waitKey(0)
                target_full_path = os.path.join(target_path, d, file)
                cv2.imwrite(target_full_path, image)
            except Exception:
                continue
                
                
if __name__ == "__main__":
    crop_font_image(args.dir_path, args.target_path)
