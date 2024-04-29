import torch
import torch.distributed as dist

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')

x = torch.randn(10, 4).cuda()

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=1,
                        rank=0)

torch.distributed.broadcast_multigpu(x, 0)

print(x.cuda(cuda1))
