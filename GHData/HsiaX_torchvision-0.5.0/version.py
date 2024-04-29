__version__ = '0.5.0'
git_version = '85b8fbfd31e9324e64e24ca25410284ef238bcb3'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
