import os
import time
from threading import Thread


# ##########nhuk#################################### multi-thread ;daemon threads
# def run(n):
#     print('task', n)
#     time.sleep(1)
#     print('2s')
#     time.sleep(1)
#     print('1s')
#     time.sleep(1)
#     print('0s')
#     time.sleep(1)
#
#
# if __name__ == '__main__':
#     t1 = Thread(target=run, args=('t1',))  # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
#     t2 = Thread(target=run, args=('t2',))
#     t1.setDaemon(True)
#     t2.setDaemon(True)
#     t1.start()
#     t2.start()
#     time.sleep(2)
# ##########nhuk####################################


##########nhuk#################################### multi-thread
# I/0密集型任务
def work():
    time.sleep(2)
    print("===>", file=open("tmp.txt", "a"))


if __name__ == "__main__":
    l = []
    print("本机为", os.cpu_count(), "核 CPU")  # 本机为4核
    start = time.time()

    for i in range(400):
        p = Thread(target=work)  # 多线程
        l.append(p)
        p.start()
    for p in l:
        p.join()
    stop = time.time()
    print("I/0密集型任务，多线程耗时 %s" % (stop - start))
##########nhuk####################################


##########nhuk#################################### multi-processing
# from multiprocessing import Process
# import os, time
#
# #I/0密集型任务
# def work():
#     time.sleep(2)
#     print("===>", file=open("tmp.txt", "w"))
#
# if __name__ == "__main__":
#     l = []
#     print("本机为", os.cpu_count(), "核 CPU")  # 本机为4核
#     start = time.time()
#     for i in range(400):
#         p = Process(target=work)  # 多进程
#         l.append(p)
#         p.start()
#     for p in l:
#         p.join()
#     stop = time.time()
#     print("I/0密集型任务，多进程耗时 %s" % (stop - start))
##########nhuk####################################
