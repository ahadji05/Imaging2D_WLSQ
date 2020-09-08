
import sys
import os

test_src_path = os.path.dirname(__file__)
sys.path.append(test_src_path + '/../cpp_src')
from interface import *

import numpy as np
import unittest
import random

sdelta = 1e-5

#--------------------------

class Testmkl_linalg(unittest.TestCase):
    
    def test_src_C_c64dot(self):
        #use 100 elements in each vector
        N = 100

        #create randomly vector A (N)
        Acl = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
        ptrA = Acl.ctypes.data_as(POINTER(ndpointer(dtype=np.complex64)))

        #create randomly vector B (N)
        Bcl = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
        ptrB = Bcl.ctypes.data_as(POINTER(ndpointer(dtype=np.complex64)))
        
        # do numpy dot product
        npdotresult = np.dot(Acl,Bcl)

        # do mkldot product
        mkldotresult = np.zeros(1, dtype=np.complex64)
        ptr = mkldotresult.ctypes.data_as(POINTER(ndpointer(np.complex64)))
        mkldot(ptrA, ptrB, N, ptr)
        
        # make comparisons
        self.assertAlmostEqual(npdotresult.real, mkldotresult[0].real, delta=sdelta)
        self.assertAlmostEqual(npdotresult.imag, mkldotresult[0].imag, delta=sdelta)


    def test_src_C_c64matvec(self):
        
        #config dimensions
        rows = int(6)
        colms = int(15)
        N = rows*colms

        #create random array A (rows x colms)
        A = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
        A.imag = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.float32)
        A = np.reshape(A,(rows,colms))

        #create random vecor B (colms)
        B = np.array(np.random.uniform(low=-2, high=2, size=(colms,)).astype(np.complex64))

        #allocate output vector (colms)
        Cmkl = np.zeros(rows, dtype=np.complex64)
        Cnp = np.zeros(rows, dtype=np.complex64)

        # do numpy matrix-vector multiplication
        Cnp = np.matmul(A,B)

        # do mkl matrix-vector multiplication
        mklmatvec(A, B, rows, colms, Cmkl)

        #compare all output results one-by-one
        for i in range(rows):
            self.assertAlmostEqual(Cnp[i].real, Cmkl[i].real, delta=sdelta)
            self.assertAlmostEqual(Cnp[i].imag, Cmkl[i].imag, delta=sdelta)
        

class Testmkl_fft(unittest.TestCase):

    def test_src_C_c64fft1dforw(self):

        list_of_N = [10, 33, 155]
        for N in list_of_N:

            A = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
            A.imag = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.float32)
            Cmkl = A

            C = np.fft.fft(A)
            
            c64fft1dforw(Cmkl, N)

            #compare all output results one-by-one
            for i in range(N):
                self.assertAlmostEqual(C[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(C[i].imag, Cmkl[i].imag, delta=sdelta)

    def test_src_C_c64fft1dback(self):
        
        list_of_N = [10, 33, 155]
        for N in list_of_N:
            
            A = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.complex64)
            A.imag = np.random.uniform(low=-2, high=2, size=(N,)).astype(np.float32)
            Cmkl = A

            C = np.fft.ifft(A)

            c64fft1dback(Cmkl, N)

            #compare all output results one-by-one
            for i in range(N):
                self.assertAlmostEqual(C[i].real, Cmkl[i].real, delta=sdelta)
                self.assertAlmostEqual(C[i].imag, Cmkl[i].imag, delta=sdelta)

class Test_extrapolation_cpu(unittest.TestCase):

    def test_cpp_src_extrapolate_1(self):

        nz = 1
        nextrap = 1
        nw = 2
        nt = nw
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+1j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        image = np.zeros((nz,nx), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,0,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,0,:] = np.array([freq[i], freq[i], freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1-1j,2-2j,3-3j,4-4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        extrapolate(nextrap, nz, nx, nw, nt,\
            M, op_forw, op_back, pulse_forw, pulse_back, image)

        self.assertAlmostEqual(image[0,0], 968.0, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 0:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_cpp_src_extrapolate_2(self):

        nz = 1
        nextrap = 1
        nw = 2
        nt = nw
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        image = np.zeros((nz,nx), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,2,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,2,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        extrapolate(nextrap, nz, nx, nw, nt,\
            M, op_forw, op_back, pulse_forw, pulse_back, image)

        self.assertAlmostEqual(image[0,2], -3690, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 2:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_cpp_src_extrapolate_3(self):
    
        nz = 1
        nextrap = 1
        nw = 2
        nt = nw
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        image = np.zeros((nz,nx), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,3,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        extrapolate(nextrap, nz, nx, nw, nt,\
            M, op_forw, op_back, pulse_forw, pulse_back, image)

        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_cpp_src_extrapolate_4(self):
        
        nz = 1
        nextrap = 1
        nw = 2
        nt = nw
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        image = np.zeros((nz,nx), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,2,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,2,:] = np.array([-freq[i], -freq[i], -freq[i]])
            op_forw[0,i,3,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        extrapolate(nextrap, nz, nx, nw, nt,\
            M, op_forw, op_back, pulse_forw, pulse_back, image)

        self.assertAlmostEqual(image[0,2], -3690, delta=sdelta)
        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3 and i != 2:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)
    
    def test_cpp_src_extrapolate_5(self):
        
        nz = 2
        nextrap = 2
        nw = 2
        nt = nw
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        image = np.zeros((nz,nx), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for l in range(nextrap):
            for i in range(nw):
                op_forw[l,i,3,:] = np.array([freq[i], freq[i], freq[i]])
                op_back[l,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        extrapolate(nextrap, nz, nx, nw, nt,\
            M, op_forw, op_back, pulse_forw, pulse_back, image)

        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)
        self.assertAlmostEqual(image[1,3], 7298, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)
                self.assertAlmostEqual(0.0, image[1,i], delta=sdelta)

