# Imaging2D_WLSQ
## Imaging with 2D wavefields using Weighted Least SQuares (WLSQ) extrapolation.

### Description of the code

The code was developed initially in Python programming language and all computationaly expensive parts are
implemented in C++ and C++-CUDA to allow better optimizations. You can run this software both in a CPU or an
NVIDIA GPU. The code is still under development for optimization purpose however it is alreafy fully functional 
and tested to produce correct results. The code is oriented to run in Linux operating systems, locally or in remote
servers.

Python works as the glue code that utilizes useful libraries such as *numpy*, *scipy*, *skimage*, *unittest* etc.
that favor developement productivity and at the same provide sufficient perfromance. For even higher performane we
use the module *ctypes* to interface with C++ or C++-CUDA compiled code. We suggest to intall Python using the Anaconda
framework.

- Python : [Anaconda_for_linux] (https://docs.anaconda.com/anaconda/install/linux/)

To be able to compile the code you need to have installed 
Intel's Math Kernel Library (MKL) and CUDA, both are available for free download.

- MKL : [MKL] (https://software.intel.com/en-us/mkl/choose-download)
- CUDA : [CUDA] (https://developer.nvidia.com/cuda-downloads)

### How to run the code

To run this code you need first to compile the C++ source code which is present in the directory ***/cpp src/*** using 
the Makefile in the same directory. To do so open the Makefile and specify the location of intel mkl directory.

In example:

INTEL = /opt/intel

MKLROOT = $(INTEL)/mkl

When you manage to compile succesfully you should see the following dynamic shared libraries in the same directory.

- extrapolation_cpu.so
- extrapolation_gpu.so
- extrapolation_revOp_gpu.so

The first library provides implementation of extrapolation and imaging on CPU. The second and third do so for GPU.
The third libray is the latest one and generally performs faster. We suggest to use this (in case you want to run on a GPU)!

The selection is done at run-time in the python script ***main.py*** according to the user's choice given
as command line parameter.

In example, to utilize the first library (CPU) the command is like:

- python main.py demo-data/velmod.csv demo-data/problemSetup.txt demo-data/seismicShots/ demo-result host

*note the last parameter "host"

to use the second library (naive GPU implementation) replace "host" with "device", and to use the third replece
with "device_revOp":

-  python main.py demo-data/velmod.csv demo-data/problemSetup.txt demo-data/seismicShots/ demo-result device_revOp

