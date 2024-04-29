"""
说明：test2笔记
"""
"""
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
参数：
    size(int…) --定义输出张量形状的整数序列。可以是数量可变的参数，也可以是列表或元组之类的集合。
关键词：
    out(Tensor, optional) --输出张量
    dtype(torch.dtype, optional) --返回张量所需的数据类型。默认:如果没有，使用全局默认值
    layout(torch.layout, optional) --返回张量的期望布局。默认值:torch.strided
    device(torch.device, optional) --返回张量的所需 device。默认:如果没有，则使用当前设备作为默认张量类型.(CPU或CUDA)
    requires_grad(bool, optional) –autograd是否应该记录对返回张量的操作(说明当前量是否需要在计算中保留对应的梯度信息)。默认值:False。
"""

"""
https://blog.csdn.net/panbaoran913/article/details/123137688
梯度归零：
    首先了解pytorch的机制。在pytorch中有前向计算图和反向计算图两个独立的机制。并且pytorch会默认对梯度进行累加。
    默认累加的好处是当在多任务中对前面共享部分的tensor进行了多次计算操作后，调用不同任务loss的backward，那些tensor的梯度会自动累加，
    缺点是当你不想先前的梯度影响到当前梯度的计算时需要手动清零 。

——————————————————————————————————————————————————————————————————————————————————————

"""


