
import sys
import os

test_src_path = os.path.dirname(__file__)
sys.path.append(test_src_path + '/../cpp_src')

import numpy as np
import unittest
import random

import time

sdelta = 1e-5

#--------------------------
from interface_cuda import extrapolate
from interface_cuda import extrapolate_revOp
from interface_cuda import extrapolate_sh_revOp

class TestDeviceExtrapolation(unittest.TestCase):
    
    def test_cuda_extrapolation_rearranged_operators_1(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,1,v] = 1+1j
                    op[l,i,1,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(pulse[1,1].real, 10.0, delta=sdelta)
        self.assertAlmostEqual(pulse[1,1].imag, -10.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,1].real, 27.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,1].imag, 25.0, delta=sdelta)
        self.assertAlmostEqual(image[0,1], 20.0 , delta=sdelta)

    def test_cuda_extrapolation_rearranged_operators_2(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,1,v] = 1+1j
                    op[l,i,1,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[0,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(pulse[0,1].real, 10.0, delta=sdelta)
        self.assertAlmostEqual(pulse[0,1].imag, -10.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,1].real, 27.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,1].imag, 25.0, delta=sdelta)
        for i in range(nz):
            for j in range(nx):
                self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)

    def test_cuda_extrapolation_rearranged_operators_3(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(pulse[1,2].real, 10.0, delta=sdelta)
        self.assertAlmostEqual(pulse[1,2].imag, -10.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].real, 27.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].imag, 25.0, delta=sdelta)
        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)

    def test_cuda_extrapolation_rearranged_operators_4(self):

        ns = 1
        nz = 5
        nextrap = 2
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(pulse[1,2].real, 0.0, delta=sdelta)
        self.assertAlmostEqual(pulse[1,2].imag, -20.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].real, 2.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].imag, 52.0, delta=sdelta)
        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    if i != 1 and j != 2:
                        self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)
        self.assertAlmostEqual(image[1,2], -1040.0 , delta=sdelta)

    def test_cuda_extrapolation_rearranged_operators_5(self):

        ns = 1
        nz = 10
        nextrap = 2
        nf = 20
        nt = nf
        nx = 16
        M = 5
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(pulse[1,2].real, 0.0, delta=sdelta)
        self.assertAlmostEqual(pulse[1,2].imag, -20.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].real, 2.0, delta=sdelta)
        self.assertAlmostEqual(pulse2[1,2].imag, 52.0, delta=sdelta)
        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    if i != 1 and j != 2:
                        self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)
        self.assertAlmostEqual(image[1,2], -1040.0 , delta=sdelta)

# --------------------------------------------------------------
# test GPU extrapolation and imaging without rearranged operators

    def test_cuda_extrapolation_operators_1(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,1,v] = 1+1j
                    op[l,i,1,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        self.assertAlmostEqual(image[0,1], 20.0 , delta=sdelta)

    def test_cuda_extrapolation_operators_2(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,1,v] = 1+1j
                    op[l,i,1,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[0,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        for i in range(nz):
            for j in range(nx):
                self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)

    def test_cuda_extrapolation_operators_3(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)

    def test_cuda_extrapolation_operators_4(self):

        ns = 1
        nz = 5
        nextrap = 2
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    if i != 1 and j != 2:
                        self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)
        self.assertAlmostEqual(image[1,2], -1040.0 , delta=sdelta)

    def test_cuda_extrapolation_operators_5(self):

        ns = 1
        nz = 10
        nextrap = 2
        nf = 20
        nt = nf
        nx = 16
        M = 5
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,2,v] = 1+1j
                    op[l,i,2,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,1:5] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,1:5] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        extrapolate(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        for i in range(nz):
            for j in range(nx):
                if i != 0 and j != 2:
                    if i != 1 and j != 2:
                        self.assertAlmostEqual(image[i,j], 0.0 , delta=sdelta)
        
        self.assertAlmostEqual(image[0,2], 20.0 , delta=sdelta)
        self.assertAlmostEqual(image[1,2], -1040.0 , delta=sdelta)

#-----------------------------------------------------------------

    def test_cuda_extrapolation_sh_rearranged_1(self):

        ns = 1
        nz = 5
        nextrap = 1
        nf = 2
        nt = nf
        nx = 6
        M = 2
        length_M = 2*M+1

        image = np.zeros((nz,nx), dtype=np.float32)

        op = np.zeros((nextrap, nf, nx, length_M), dtype=np.complex64)
        op2 = np.zeros_like(op)
        for l in range(nextrap):
            for i in range(nf):
                for v in range(3+M):
                    op2[l,i,1,v] = 1+1j
                    op[l,i,1,v] = 1-1j
        
        pulse = np.zeros((nf, nx), dtype=np.complex64)
        pulse[1,0:4] = np.array([1+0j,2+0j,3+0j,4+0j], dtype=np.complex64)
        pulse2 = np.zeros_like(pulse)
        pulse2[1,0:4] = np.array([5+1j,6-2j,7+0j,8+0j], dtype=np.complex64)

        print('\n')
        print(pulse)
        print(pulse2)

        extrapolate_sh_revOp(ns, nextrap, nz, nt, nf, nx, M, op, pulse, op2, pulse2, image)

        print(pulse)
        print(pulse2)
        print(image)
