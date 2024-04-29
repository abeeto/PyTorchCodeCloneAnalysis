import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description="NGD Benchmarking Result Processor")
parser.add_argument(
    "--DATA_SAVE_FOLDER", "-s", type=str, default="/gpfs/fs0/data/DeepLearning/sabedin/Data/ngd-benchmark/sample-data",
    required=False, help="folder to save results"
)
parser.add_argument("--NUM_SAMPLES", "-n", type=int,
                    default=8 * (5 + 50), required=False,
                    help="Dict")
args = parser.parse_args()


def main():
    mean = 0
    variance = 1
    for i in range(args.NUM_SAMPLES):
        # Generate random normal distribution with mean 0 and variance 1 just like in torch.randn
        data = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(3, 224, 224))
        # File Save Path
        file_save_path = (os.path.join(args.DATA_SAVE_FOLDER, str(i).zfill(6) + ".npy"))
        # Save File
        np.save(file_save_path, data)


if __name__ == "__main__":
    print('Saving', args.NUM_SAMPLES, 'samples in', args.DATA_SAVE_FOLDER)
    main()
