import numpy as np
import argparse
import struct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_root")
    parser.add_argument("img_height",type=int)
    parser.add_argument("img_width",type=int)

    args = parser.parse_args()

    visualize_idxs = np.fromfile(args.config_root+"visualize_idxs.bin",np.int32).reshape([-1,2])
    
    img_config = struct.pack("ii",args.img_height,args.img_width)

    with open(args.config_root+"visualize_config_torch.bin","wb") as pf:
        pf.write(img_config)
        visualize_idxs.tofile(pf)