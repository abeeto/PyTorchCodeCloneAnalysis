__version__ = '0.7.0a0+df8d776'
git_version = 'df8d7767d0f47f7e6869b9d2f92a902c5cb6e03d'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
