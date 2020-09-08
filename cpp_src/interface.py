
import sys
import os
from ctypes import CDLL, POINTER, c_int, c_void_p, c_float, CFUNCTYPE
from numpy.ctypeslib import ndpointer
import numpy as np

#------------------------
interface_path = os.path.dirname(__file__)

mkl_linalg = CDLL(interface_path+'/mkl_linalg.so')
mkl_fft = CDLL(interface_path + '/mkl_fft.so')
extrapolation_cpu = CDLL(interface_path + '/extrapolation_cpu.so')

#------------------------
#   mkl_linalg
#------------------------
mkldot = mkl_linalg.c64dot
mkldot.restype = c_void_p
mkldot.argtypes = [POINTER(ndpointer(np.complex64)), 
                   POINTER(ndpointer(np.complex64)),
                   c_int,
                   POINTER(ndpointer(np.complex64))]

mklmatvec = mkl_linalg.c64matvec
mklmatvec.restype = c_void_p
mklmatvec.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                      ndpointer( dtype=np.complex64, flags=("C","A") ),
                      c_int,
                      c_int,
                      ndpointer( dtype=np.complex64, flags=("C","A") )]

#--------------------------
#   mkl_fft
#--------------------------
c64fft1dforw = mkl_fft.c64fft1dforw
c64fft1dforw.restype = c_void_p
c64fft1dforw.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                         c_int]

c64fft1dback = mkl_fft.c64fft1dback
c64fft1dback.restype = c_void_p
c64fft1dback.argtypes = [ndpointer( dtype=np.complex64, flags=("C","A") ),
                         c_int]

#--------------------------
#   extrapolation
#--------------------------
extrapolate = extrapolation_cpu.extrapolate
extrapolate.restype = c_void_p
extrapolate.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") )]


