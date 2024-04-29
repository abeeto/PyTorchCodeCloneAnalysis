import torch
import time

#assert torch.cuda.device_count() >= 2

class Benchmark():  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))

def run(x):
    for _ in range(20000):
        y = torch.mm(x, x)
        
x_cpu = torch.rand(size=(1000, 1000), device='cpu')
x_gpu1 = torch.rand(size=(1000, 1000), device='cuda:0')       

with Benchmark('Run on CPU.'):
    run(x_cpu)
        
with Benchmark('Run on GPU1.'):
    run(x_gpu1)
    torch.cuda.synchronize()

# Run on CPU. time: 226.5659 sec
# Run on GPU1. time: 38.9012 sec

#with Benchmark('Then run on GPU2.'):
#    run(x_gpu2)
#    torch.cuda.synchronize()
#    
#with Benchmark('Run on both GPU1 and GPU2 in parallel.'):
#    run(x_gpu1)
#    run(x_gpu2)
#    torch.cuda.synchronize()

