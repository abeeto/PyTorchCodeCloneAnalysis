import os, sys
import cdsw

NUM_WORKERS = 2
CMD = '''mpirun -np {} -H {} -bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib -mca plm_rsh_args '-p 2222 -o StrictHostKeyChecking=no' -mca btl_tcp_if_include eth0 \
python3 dist-torch-hvd.py'''

workers = cdsw.launch_workers(n=NUM_WORKERS, cpu=2, memory=4, nvidia_gpu=1,
                              code="import time; time.sleep(365*24*3600)")
print('Starting workers ...')
worker_ids = [worker["id"] for worker in workers]
running_workers = cdsw.await_workers(worker_ids,
                              wait_for_completion=False,
                              timeout_seconds=120)
worker_ips = [worker["ip_address"] for worker in \
                              running_workers["workers"]]
print('Workers:', worker_ips)

if len(running_workers) == NUM_WORKERS:
    hosts_str = ",".join([worker_ip+":1" for worker_ip in worker_ips])
    cmd = CMD.format(len(worker_ips), hosts_str)
    # cmd = "horovodrun -np {} -H {} -p 2222 python3 dist-torch-hvd.py 2>&1".format(
    #                              len(worker_ips),
    #                              hosts_str)
    print('Preparing to run: ' + cmd)
    os.system(cmd)
    print('DONE')
    #cdsw.stop_workers()
else:
    print('Errors when starting workers. Exits.')
    sys.exit(1)
