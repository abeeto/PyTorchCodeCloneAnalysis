import time
import os

last_time = time.time()
t = 0
while True:
    elapsed_time = time.time() - last_time
    if elapsed_time > 5:
        os.system('./autocap.sh')
        last_time = time.time()
