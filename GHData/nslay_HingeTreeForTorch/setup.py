# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# November 2020
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

from setuptools import setup, Extension
import torch.cuda
from torch.utils import cpp_extension

sourceFiles = [ 'hingetree.cpp', 'hingetrie.cpp', 'ImageToMatrix.cpp', 'hingetree_conv.cpp', 'hingetree_fused_linear.cpp', 'expand.cpp' ]
extraCflags = [ '-O2' ]
extraCudaFlags = [ '-O2' ]

if torch.cuda.is_available():
  sourceFiles.append('hingetree_gpu.cu')
  sourceFiles.append('ImageToMatrix_gpu.cu')
  sourceFiles.append('hingetree_conv_gpu.cu')
  sourceFiles.append('hingetree_fused_linear_gpu.cu')
  extraCflags.append('-DWITH_CUDA=1')
  extraCudaFlags.append('-DWITH_CUDA=1')

  setup(name='hingetree_cpp',
      ext_modules=[cpp_extension.CUDAExtension(name = 'hingetree_cpp', sources = sourceFiles, extra_compile_args = {'cxx': extraCflags, 'nvcc': extraCudaFlags})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
else:
  setup(name='hingetree_cpp',
      ext_modules=[cpp_extension.CppExtension(name = 'hingetree_cpp', sources = sourceFiles, extra_compile_args = {'cxx': extraCflags, 'nvcc': extraCudaFlags})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
