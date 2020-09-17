
import sys

sys.path.append('./../cpp_src')

import numpy as np
import time

from interface_cuda import extrapolate_revOp
from interface_cuda import extrapolate

ns = 1
nz = 200
nextrap = 195
nf = 75
nt = nf
nx = 501
M = 25
length_M = 2*M+1

image = np.zeros((nz,nx), dtype=np.float32)
op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
for l in range(nextrap):
    for i in range(nf):
        for v in range(3+M):
            op[l,i,0,v] = 1+0j

pulse = np.zeros((nf, nx), dtype=np.complex64)
pulse[0,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
pulse[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

t0 = time.time()
extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op, pulse, image)
tf = time.time()
print("GPU v2:",tf - t0,"seconds.")
