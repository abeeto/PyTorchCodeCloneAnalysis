#!/usr/bin/env python


from distutils.version import StrictVersion

import argparse
import hpccm


parser = argparse.ArgumentParser(description='HPCCM Torch-OpenACC')
parser.add_argument('--gcc_version', type=str, default='11',
                    help='GNU Toolchain version (default: 11)')

#parser.add_argument('--cuda', type=str, default='9.1',
#                    help='CUDA version (default: 9.1)')
#parser.add_argument('--format', type=str, default='docker',
#                    choices=['docker', 'singularity'],
#                    help='Container specification format (default: docker)')
#parser.add_argument('--ompi', type=str, default='3.1.2',
#                    help='OpenMPI version (default: 3.1.2)')


args = parser.parse_args()


### Create Stage
Stage0 = hpccm.Stage()



Stage0 += hpccm.primitives.baseimage(image='nvcr.io/nvidia/pytorch:22.04-py3')

#nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04'
#docker://nvcr.io/nvidia/pytorch:22.04-py3


# -----------------JUPYTER: BEGIN
#CHECAR https://github.com/NVIDIA/hpc-container-maker/blob/master/recipes/jupyter/jupyter.py
#stage += hpccm.building_blocks.conda(packages=['ipython', 'jupyter'],
#                                          requirements=args.requirements)

### Make the port accessible (Docker only)
#stage += hpccm.primitives.raw(docker='EXPOSE 8888')

### Add the notebook itself
#stage += hpccm.primitives.copy(src=args.notebook, dest='/notebook/',
#                           _mkdir=True)


#stage += hpccm.primitives.shell(commands=[
#    'echo "#!/bin/bash\\nsource /usr/local/anaconda/bin/activate {}\\njupyter notebook --ip 0.0.0.0 --no-browser --notebook-dir /notebook --allow-root" > /usr/local/bin/entrypoint.sh'.format(env),
#    'chmod a+x /usr/local/bin/entrypoint.sh'])
#stage += hpccm.primitives.runscript(
#    commands=['/usr/local/bin/entrypoint.sh'])
# -----------------JUPYTER: END


#GNU_TOOLCHAIN = hpccm.building_blocks.gnu(source=True,
#                     version=args.gcc_version,
#                     fortran=False,
#                     extra_repository=True,
#                     openacc=True)
#                     configure_opts=["--disable-bootstrap",
#                                    "--disable-multilib"])
#Stage0 += GNU_TOOLCHAIN
#Stage0 += mlnx_ofed(version='5.4-1.0.3.0')
#Stage0 += knem(ldconfig=True)


#PGI_TOOLCHAIN = pgi(eula=True, mpi=True, )

#PACKAGES = hpccm.building_blocks.packages(ospackages=['make', 'wget', 'cmake'])
#PACKAGES = hpccm.building_blocks.packages(ospackages=['wget', 'gcc-11-offload-nvptx'])
#Stage0 += PACKAGES


#if UBUNTU
#APT = packages(ospackages=['apt-utils'])
#Stage0 += APT



#Stage0 += gdrcopy(version='2.3', ldconfig=True)

#NCCL = nccl(version='2.11.4-1', cuda='11.4', environment=True)
#Stage0 += NCCL

NVHPC = hpccm.building_blocks.nvhpc( eula=True,
               cuda='11.6',
               version='22.3',
               cuda_multi=False,
               mpi=True,
               environment=True,
               extended_environment=True,
               ospackages=[ 'gcc-offload-nvptx',\
                            'bc', \
                            'debianutils',\
                            'g++', \
                            'gcc', \
                            'gfortran', \
                            'libatomic1', \
                            'libnuma1', \
                            'openssh-client', \
                            'wget' ],
               tarball=True)
Stage0 += NVHPC


#CONDA = conda(eula=True, packages=['numpy'], channels=['conda-forge', 'nvidia', 'pytorch'])
#Stage0 += CONDA

#PIP = pip(packages=['hpccm'], pip='pip3')
#Stage0 += PIP


#HDF5 = hdf5(toolchain=NVHPC.toolchain, check=True, configure_opts=['--enable-cxx',
#                '--enable-profiling=yes'])
#Stage0 += HDF5 



#Stage0 += ucx(version='1.12.0', ldconfig=True)
#Stage0 += slurm_pmi2(toolchain=NVHPC.toolchain)
#Stage0 += openmpi(version=OMPI_VERSION, ldconfig=True, pmi=f"/usr/local/slurm-pmi2", pmix="internal", toolchain=NVHPC.toolchain)


#Stage0 += shell(commands=[
#    'export PATH=/usr/local/anaconda/bin:/usr/local/anaconda/condabin:$PATH',
#    'conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch'])
    
    
    
    
print(Stage0)
 
