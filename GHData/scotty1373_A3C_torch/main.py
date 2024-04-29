# -*- coding:utf-8 -*-
import sys
import os
import time
import torch
from utils_tools.A3C import AC_Net, worker
import torch.multiprocessing as mp
from utils_tools.utils import smooth_tsplot


STATE_DIM = 3
ACTION_BOUND = 2
THREAD_NUM = 16


if __name__ == '__main__':
    global_net = AC_Net(STATE_DIM, ACTION_BOUND)
    # global_net.AC.share_memory()
    g_opt = torch.optim.Adam(params=global_net.AC.parameters(), lr=1e-4)

    # 创建共享内存和跨进程通信队列
    global_ep = mp.Value('i', 0)    # C_Type int类型共享内训
    global_r = mp.Value('f', 0)

    # 将训练数据从不同的worker中通过队列传出
    res_queue = [mp.Queue() for que_idx in range(THREAD_NUM)]

    # 从自定义进程类worker中创建子线程
    p_list = [worker(f'{idx}_worker', global_net, g_opt, global_ep, global_r, res_queue[idx], idx) for idx in range(THREAD_NUM)]
    [p.start() for p in p_list]

    res = {str(work_id): [] for work_id in range(THREAD_NUM)}
    while True:
        for x in res_queue:
            r_ep = x.get()
            if r_ep:
                res[str(r_ep[0])].append(r_ep[1])
            else:
                if not r_ep:
                    break
                p_list[r_ep[0]].join()
                res_queue.pop(r_ep[0])
        if not r_ep:
            break

    smooth_tsplot(res, thread_num=THREAD_NUM)

    time.time()

