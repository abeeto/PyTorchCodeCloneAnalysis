# coding=utf-8

import warnings
try:
    import pynvml  # nvidia-ml provides utility for NVIDIA management

    HAS_NVML = True
except:
    HAS_NVML = False

import sys
import time


def auto_select_gpu(threshold=1000, show_info=True):
    '''
    Select gpu which has largest free memory
    :param threshold: MB
    :param show_info:
    :return:
    '''

    def KB2MB(memory):
        return memory / 1024.0 / 1024.0

    if HAS_NVML:
        gpu_memories = {}

        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        largest_free_mem = 0
        largest_free_idx = 0
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_memories[str(i)] = KB2MB(info.free)
            if info.free > largest_free_mem:
                largest_free_mem = info.free
                largest_free_idx = i

        pynvml.nvmlShutdown()
        largest_free_mem = KB2MB(largest_free_mem)
        print(
            f'================== Large_free_size: {largest_free_mem}===============')

        if largest_free_mem > threshold:
            info = '==================Using GPU {} with free memory {}MB ===============' \
                   ''.format(largest_free_idx, largest_free_mem)
            if show_info:
                print(info)
            return '{}'.format(largest_free_idx)
        else:
            warnings.warn('No valid GPU can use !')
            return ''
    else:
        info = 'pynvml is not installed, automatically select gpu is disabled!'
        warnings.warn(info)
        sys.exit(0)


def inquire_gpu(interval=1.0, show_info=True):
    """
    no valid gpu hang out
    :param interval: minute
    :return:
    """
    index = auto_select_gpu()
    interval = int(interval * 60)
    while index == '':
        time.sleep(interval)
        index = auto_select_gpu()
        if show_info:
            print('Sleep', time.strftime('%Y-%m-%d_%H:%M:%S'))

    return index


if __name__ == '__main__':
    print('select', auto_select_gpu(10000))
