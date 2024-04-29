__version__ = '0.11.3'
git_version = '05eae32f9663bbecad10a8d367ccbec50130e2f5'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
