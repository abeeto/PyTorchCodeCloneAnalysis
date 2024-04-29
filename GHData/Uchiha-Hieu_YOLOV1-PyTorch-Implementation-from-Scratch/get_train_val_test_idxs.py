import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--img_root",type=str,help="Path to Image Directory")
    parser.add_argument("-o","--output_root",type=str,help="Path to directory to save txt files")
    return parser.parse_args()

def main(args):
    img_root = args.img_root
    output_root = args.output_root
    if os.path.isdir(output_root) == False:
        os.mkdir(output_root)
    num_imgs = len(os.listdir(img_root))
    idxs = np.arange(num_imgs)
    train_idxs,val_test_idxs = train_test_split(idxs,test_size=0.4)
    val_idxs,test_idxs = train_test_split(val_test_idxs,test_size=0.5)
    np.save(os.path.join(output_root,"train.npy"),train_idxs)
    np.save(os.path.join(output_root,"val.npy"),val_idxs)
    np.save(os.path.join(output_root,"test.npy"),test_idxs)
    print("Num imgs : ",num_imgs)
    print("Train imgs : ",len(train_idxs))
    print("Val imgs :",len(val_idxs))
    print("Test imgs : ",len(test_idxs))

if __name__ == '__main__':
    args = arguments()
    main(args)
    # python get_train_val_test_idxs.py -i "D:\\datasets\\geo_shapes\\train" -o "train_val_test_idxs"