# -*- coding: utf-8 -*-
# created by makise, 2022/4/9


import time
import pebble
from concurrent.futures import TimeoutError

def time_sleep(sleep):
    print("sleep %s" % sleep)
    time.sleep(sleep)
    return sleep


def while_sleep():
    print("Start while sleep")
    a = 0
    while True:
        a += 1


def main():
    with pebble.ProcessPool(1) as pool:
        for i in range(2):
            future = pool.schedule(time_sleep, (10,))
            try:
                print("future.result() = %s" % future.result(timeout=2))
            except TimeoutError:
                future.cancel()
                print("Timeout")


if __name__ == "__main__":
    main()