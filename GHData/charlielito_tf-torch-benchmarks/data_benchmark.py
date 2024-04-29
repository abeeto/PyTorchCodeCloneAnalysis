import os
import time
from math import ceil

import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tqdm import tqdm

from pytorch_imp.data import get_dataloader, IterDataset
from tensorflow_imp.data import get_dataset


def main():
    data_dir = "gs://tf-vs-torch/test-data/images/pokemon_jpg"
    data_dir = "data/pokemon_jpg"
    image_paths = [
        os.path.join(data_dir, file) for file in tf.io.gfile.listdir(data_dir)
    ]

    # image_paths = image_paths[:300]
    # image_paths = image_paths + image_paths

    batch_size = 32
    num_workers = 8
    prefetch_factor = 2
    epochs = 10
    time_per_step = 0.01

    steps_per_epoch = ceil(len(image_paths) / batch_size)
    cache_dir = None

    # tensorflow
    tf_dataset = get_dataset(
        image_paths,
        load_num_parallel_calls=num_workers,
        batch_size=batch_size,
        shuffle=True,
        cache_dir=cache_dir,
        repeat=False,
        prefetch_factor=prefetch_factor,
    )

    init_tf = time.time()
    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch: "):
        for step, batch in tqdm(enumerate(tf_dataset), total=steps_per_epoch, desc="Step: "):
            time.sleep(time_per_step)
    total_tf = time.time() - init_tf
    print(f"Total time tensorflow: {total_tf} \n")
    # exit()

    # pytorch
    torch_dataset = get_dataloader(
        image_paths,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=prefetch_factor,
    )

    init_torch = time.time()
    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch: "):
        for step, batch in tqdm(enumerate(torch_dataset), total=steps_per_epoch, desc="Step: "):
            time.sleep(time_per_step)
    total_torch = time.time() - init_torch
    print(f"Total time pytorch: {total_torch}")

    # tf.data + pytorch dataloader
    tf_dataset2 = get_dataset(
        image_paths,
        load_num_parallel_calls=num_workers,
        batch_size=batch_size,
        shuffle=True,
        cache_dir=cache_dir,
        repeat=True,
        prefetch_factor=prefetch_factor,
    )
    torch_dataset2 = torch.utils.data.DataLoader(
        IterDataset(
            tf_dataset2.as_numpy_iterator(),
            length=steps_per_epoch,
        ),
        num_workers=0,
        batch_size=None,
    )

    init_torch = time.time()
    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch: "):
        for step, batch in tqdm(enumerate(torch_dataset2), total=steps_per_epoch, desc="Step: "):
            time.sleep(time_per_step)
    total_torch = time.time() - init_torch
    print(f"Total time pytorch: {total_torch}")


if __name__ == "__main__":
    main()
