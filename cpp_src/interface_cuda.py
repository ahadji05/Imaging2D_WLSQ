
import sys
import os 
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.dirname(__file__)

extrapolation_gpu = CDLL(interface_path + '/extrapolation_gpu.so')

extrapolate = extrapolation_gpu.extrapolate
extrapolate.restype = c_void_p
extrapolate.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") )]
