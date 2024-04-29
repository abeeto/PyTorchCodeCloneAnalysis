import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_space")

    args = parser.parse_args()

    tf_data = np.fromfile(args.work_space+"tf_result.bin",np.float32)
    torch_data = np.fromfile(args.work_space+"torch_result.bin",np.float32)

    print("all close:",np.allclose(tf_data,torch_data,1e-3))
    show_num = 10
    print("tf_data:",tf_data[:show_num])
    print("torch_data:",torch_data[:show_num])