from torchdata.datapipes.iter import FileLister, IterableWrapper
from torch.utils.data.datapipes.utils.decoder import imagehandler
import time

def time_decode_before_shuffle(buffer_size):
    print(f"decode before shuffle, buffer size {buffer_size}")
    dp = FileLister("caltech-101") \
    .open_files(mode="b") \
    .load_from_tar() \
    .routed_decode(imagehandler("torch")) \
    .shuffle(buffer_size=buffer_size) \

    for i, x in enumerate(dp):
        if i == 0:
            t1 = time.time()
        if i == 100:
            print("100th image: {}".format(time.time() - t1))
            break

def time_decode_after_shuffle(buffer_size):
    print(f"decode after shuffle, buffer size {buffer_size}")
    dp = FileLister("caltech-101") \
    .open_files(mode="b") \
    .load_from_tar() \
    .shuffle(buffer_size=buffer_size) \
    .routed_decode(imagehandler("torch")) \

    for i, x in enumerate(dp):
        if i == 0:
            t1 = time.time()
        if i == 100:
            print("100th image: {}".format(time.time() - t1))
            break


time_decode_before_shuffle(10)
time_decode_after_shuffle(10)
time_decode_before_shuffle(100)
time_decode_after_shuffle(100)
time_decode_before_shuffle(1000)
time_decode_after_shuffle(1000)