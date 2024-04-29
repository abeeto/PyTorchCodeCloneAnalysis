# Torch ImageHash modules
from phash import PHasher, PHasherPIL

# original image hashing module
from imagehash import phash

# misc libs
from PIL import Image
import torch
import time
from tqdm import tqdm

from utils import pil_2_tensor

NUM_ITER = 1_000
def benchmark_gpu(f, img, test_name):
    f(img) # run all operations once for cuda warm-up
    torch.cuda.synchronize() # wait for warm-up to finish

    times = []
    for e in tqdm(range(NUM_ITER), desc="running gpu benchmark"):
        start_epoch = time.time()
        f(img)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    print("{}: AVG RUNTIME:{}".format(test_name, sum(times)/NUM_ITER), "\n")
    return sum(times)/NUM_ITER

def benchmark_cpu(f, img, test_name):
    times = []
    for e in tqdm(range(NUM_ITER), desc="running cpu benchmark"):
        start_epoch = time.time()
        f(img)
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    print("{}: AVG RUNTIME:{}".format(test_name, sum(times)/NUM_ITER), "\n")
    return sum(times)/NUM_ITER


if __name__ == "__main__":
    pil_img = Image.open("fat-bird.jpg")

    # Benchmark Image Hash
    avg_time = benchmark_cpu(phash, pil_img, "CPU PHash")

    # Benchmark Torch Image Hash PIL
    # model_pil = PHasherPIL()
    # benchmark_gpu(model_pil, pil_img, "GPU PHash w/ PIL Preprocessing")

    # Benchmark Torch Image Hash
    model = PHasher()
    img_as_tensor = pil_2_tensor(pil_img)
    img_as_tensor = img_as_tensor.to("cuda")
    avg_time_gpu = benchmark_gpu(model, img_as_tensor, "GPU PHash")

    print("GPU was", (avg_time/avg_time_gpu), "faster")
