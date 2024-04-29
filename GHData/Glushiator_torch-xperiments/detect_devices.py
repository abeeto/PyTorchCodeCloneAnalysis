import torch


def _main():
    for _device_name in "cpu cuda xpu mkldnn opengl opencl ideep hip ve ort mlc xla lazy vulkan meta hpu".split():
        try:
            torch.rand(10).to(torch.device(_device_name))
            print(f"{_device_name=}")
        except RuntimeError:
            pass


if __name__ == '__main__':
    _main()
