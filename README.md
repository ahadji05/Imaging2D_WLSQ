# Imaging2D_WLSQ
## Imaging with 2D wavefields using Weighted Least SQuares (WLSQ) extrapolation.

The code was developed initially in Python programming language and all computationaly expensive parts are
implemented in C++ and C++-CUDA to allow better optimizations. You can run this software both in a CPU or an
NVIDIA GPU. The code is still under development for optimization purpose however it is alreafy fully functional 
and tested to produce correct results. The code is oriented to run in Linux operating systems, locally or in remote
servers.

Currently, the minimum requirements need to fullfil to be able to use this code, is to have installed the Python
programming language and Intel Math Kernel library (MKL).

To run this code you need to compile the source code present in directory cpp source using the Makefile
in the same directory. To be able to compile the code you need to have installed Intel MKL and CUDA.
Both are available for free download.

- MKL : [MKL](https://software.intel.com/en-us/mkl/choose-download)
- CUDA : [CUDA](https://developer.nvidia.com/cuda-downloads)



