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
use the module *ctypes* to interface with C++ or C++-CUDA compiled code. We suggest to intall Python using anaconda
framework (see provided link).

- Python : [Anaconda_for_linux] (https://docs.anaconda.com/anaconda/install/linux/)

To be able to compile the code you need to have installed 
Intel's Math Kernel Library (MKL) and CUDA, both are available for free download.

- MKL : [MKL] (https://software.intel.com/en-us/mkl/choose-download)
- CUDA : [CUDA] (https://developer.nvidia.com/cuda-downloads)

### How to run the code

To run this code you need to compile the C++ source code which is present in the directory ***/cpp src/*** using 
the Makefile in the same directory. 
