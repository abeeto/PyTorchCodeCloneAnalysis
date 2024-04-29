import time
import os

last_time1 = time.time()
last_time2 = time.time()

while True:
    elapsed_time1 = time.time() - last_time1
    elapsed_time2 = time.time() - last_time2
    if elapsed_time1 > 5:
        os.system('./autocap.sh')
        last_time1 = time.time()
    if elapsed_time2 > 10:
        os.system('./autoup2.sh')
        last_time2 = time.time()
