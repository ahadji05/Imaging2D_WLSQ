
import sys
import os 
from ctypes import CDLL, c_int, c_void_p, c_float
from numpy.ctypeslib import ndpointer
import numpy as np

interface_path = os.path.dirname(__file__)

extrapolation_gpu = CDLL(interface_path + '/extrapolation_gpu.so')
extrapolation_sh_gpu = CDLL(interface_path + '/extrapolation_sh_gpu.so')
extrapolation_sh_AoS_gpu = CDLL(interface_path + '/extrapolation_sh_AoS_gpu.so')
extrapolation_revOp_gpu = CDLL(interface_path + '/extrapolation_revOp_gpu.so')


#----------------------------------------------------------------------
#
extrapolate = extrapolation_gpu.extrapolate
extrapolate.restype = c_void_p
extrapolate.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") )]


#----------------------------------------------------------------------
#
extrapolate_sh = extrapolation_sh_gpu.extrapolate
extrapolate_sh.restype = c_void_p
extrapolate_sh.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") )]

#----------------------------------------------------------------------
#
extrapolate_sh_AoS = extrapolation_sh_AoS_gpu.extrapolate
extrapolate_sh_AoS.restype = c_void_p
extrapolate_sh_AoS.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") )]


#----------------------------------------------------------------------
#
extrapolate_revOp = extrapolation_revOp_gpu.extrapolate
extrapolate_revOp.restype = c_void_p
extrapolate_revOp.argtypes = [c_int, c_int, c_int,
                   c_int, c_int, c_int, c_int,
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.complex64, flags=("C","A") ),
                   ndpointer( dtype=np.float32, flags=("C","A") )]

                   