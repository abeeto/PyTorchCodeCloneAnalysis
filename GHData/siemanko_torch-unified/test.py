import torch
import torch_unified


def test():
    torch_cpu, torch_gpu = torch_unified.empty_unified((2, 3, 4), torch.float32)
    assert torch_cpu.device.type == "cpu"
    assert torch_gpu.device.type == "cuda"
    assert torch_cpu.data_ptr() == torch_gpu.data_ptr()
    assert torch_cpu.shape == torch_gpu.shape and torch_cpu.shape == (2, 3, 4)
    assert torch_cpu.dtype == torch.float32
    assert torch_gpu.dtype == torch.float32
