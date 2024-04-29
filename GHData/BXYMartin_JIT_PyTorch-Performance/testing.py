import threading, torch, time, pynvml

def preload_pytorch():
    torch.ones((1, 1)).cuda()

def gpu_mem_used(id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int(info.used/2**20)

def gpu_mem_used_no_cache(id):
    torch.cuda.empty_cache()
    return gpu_mem_used(id)

def peak_monitor_start():
    global peak_monitoring
    peak_monitoring = True

    # this thread samples RAM usage as long as the current epoch of the fit loop is running
    peak_monitor_thread = threading.Thread(target=peak_monitor_func)
    peak_monitor_thread.daemon = True
    peak_monitor_thread.start()

def peak_monitor_stop():
    global peak_monitoring
    peak_monitoring = False

def peak_monitor_func():
    global nvml_peak, peak_monitoring
    nvml_peak = 0
    id = torch.cuda.current_device()

    while True:
        nvml_peak = max(gpu_mem_used(id), nvml_peak)
        if not peak_monitoring: break
        time.sleep(0.001) # 1msec

def consume_gpu_ram(n): return torch.ones((n, n)).cuda()
def consume_gpu_ram_256mb(): return consume_gpu_ram(2**13)

peak_monitoring = False
nvml_peak = 0
preload_pytorch()
pynvml.nvmlInit()
id = torch.cuda.current_device()

# push the pytorch's peak gauge high up and then release the memory
z = [consume_gpu_ram_256mb() for i in range(4)] # 1GB
del z

peak_monitor_start()
nvml_before = gpu_mem_used_no_cache(id)
cuda_before = int(torch.cuda.memory_allocated()/2**20)

# should be: 256 used, 512 peaked
c1 = consume_gpu_ram_256mb()
c2 = consume_gpu_ram_256mb()
del c1

# code finished
peak_monitor_stop()
nvml_after = gpu_mem_used_no_cache(id)
cuda_after = int(torch.cuda.memory_allocated()/2**20)
cuda_peak  = int(torch.cuda.max_memory_allocated()/2**20)
print("nvml:", nvml_after-nvml_before, nvml_peak-nvml_before)
print("cuda:", cuda_after-cuda_before, cuda_peak-cuda_before)
