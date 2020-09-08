
import sys
import numpy as np
import unittest

sys.path.append('../py_src/wavefieldPropagation/')

sdelta = 1e-5

from extrapolation import c64copy2d as copy

class test_c64copy2d(unittest.TestCase):

    def test_c64copy2d(self):

        N = 25

        #generate random array
        E = np.random.uniform(low=-4, high=4, size=(N,)).astype(np.float32)

        E =  E.reshape((5,5))

        check = copy(E, 5, 5)
        
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(check[i][j], E[i][j], delta=sdelta)

        N = 42

        #generate random array
        E = np.random.uniform(low=-4, high=4, size=(N,)).astype(np.complex64)
        E.imag = np.random.uniform(low=-4, high=4, size=(N,)).astype(np.float32)

        E =  E.reshape((7,6))

        check = copy(E, 7, 6)
        
        for i in range(7):
            for j in range(6):
                self.assertAlmostEqual(check[i][j], E[i][j], delta=sdelta)

from extrapolation import extrapolate
image = extrapolate(1, 2, 4, 2, 1, \
    np.zeros((1, 2, 4, 3), dtype=np.float32), \
    np.zeros((1, 2, 4, 3), dtype=np.float32), \
    np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32), \
    np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32))

class Test_extrapolate(unittest.TestCase):
    
    def test_extrapolate_floats(self):

        nz = 1
        nextrap = 1
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1,2])

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.float32)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.float32)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,1,:] = np.array([freq[i],freq[i],freq[i]])
            op_back[0,i,1,:] = np.array([-1*freq[i],-1*freq[i],-1*freq[i]])

        pulse_forw = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)
        pulse_back = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)
        
        #CHECK 1:
        #----------------------------------------------------------------------
        #since only for spatial position 1 we have non-zero operators, only for
        #that position image value will be non-zero.
        for i in range(nx):
            if i != 1:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)
        
        #CHECK 2:
        #----------------------------------------------------------------------
        #image pixel is an element-by-element multiplication followed by reduction
        #over all frequencies of the convolutions. the two frequencies have opposite
        #signs => the image pixel must be negative in this case
        self.assertLess(image[0,1], 0)
        
        #CHECK 3
        #----------------------------------------------------------------------
        #the convolution for the forward operator of frequency 1, for the 
        #spatial location 1, is: 1*1 + 1*2 + 1*3 = 6
        #the same convolution for backward operator is: -1*1 + -1*2 + -1*3 = -6
        #FOR the second frequency and again spatial position 1 will be:
        # 2*5 + 2*6 + 2*7 = 36 and again for the backward operator will be correspondigly
        # -36 because the operator value are -2.
        #The image pixel value is an add.reduce operation = -36*36 + -6*6 = -1332
        self.assertAlmostEqual(image[0,1], -1332, delta=sdelta)

        #CHECK 4
        #----------------------------------------------------------------------
        #lets change the forward operator values:
        for i in range(nw):
            op_forw[0,i,1,:] = np.array([0,0,1])

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)
        
        #now only the last element survives for the forward operatos =>
        #the image pixel value will be: -36*7 + (-6)*3 = -270
        self.assertAlmostEqual(image[0,1], -270, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 1:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

        #CHECK 5
        #----------------------------------------------------------------------
        #now lets make the first test but now for the spatial location 0
        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.float32)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.float32)

        #only the frequencies of spatial point 0 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,0,:] = np.array([freq[i],freq[i],freq[i]])
            op_back[0,i,0,:] = np.array([-1*freq[i],-1*freq[i],-1*freq[i]])

        pulse_forw = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)
        pulse_back = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)

        #since we use padded zeros in halo regions the convolution of spatial points 0
        #have a zero term from the left halo region.
        #for the forward convolution for frequency 0 we have: 1*0 + 1*1 + 1*2 = 3
        #for the backward: -1*0 + -1*1 + -1*2 = -3
        #and for the frequency 1 we have: 2*0+2*5+2*6 = 22 and -22 correspondigly
        #the image pixel of spatial point 0 should be: -3*3 + -22*22 = -493
        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,0], -493, delta=sdelta)
        #check that all other values are zero
        for i in range(nx):
            if i != 0:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_extrapolate_complex_1(self):

        nz = 1
        nextrap = 1
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+1j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,0,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,0,:] = np.array([freq[i], freq[i], freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1-1j,2-2j,3-3j,4-4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,0], 968.0, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 0:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_extrapolate_complex_2(self):
    
        nz = 1
        nextrap = 1
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,2,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,2,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,2], -3690, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 2:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_extrapolate_complex_3(self):
        
        nz = 1
        nextrap = 1
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,3,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_extrapolate_complex_4(self):
        
        nz = 1
        nextrap = 1
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for i in range(nw):
            op_forw[0,i,2,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,2,:] = np.array([-freq[i], -freq[i], -freq[i]])
            op_forw[0,i,3,:] = np.array([freq[i], freq[i], freq[i]])
            op_back[0,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,2], -3690, delta=sdelta)
        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3 and i != 2:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)

    def test_extrapolate_complex_5(self):
        
        nz = 2
        nextrap = 2
        nw = 2
        nx = 4
        M = 1
        length_M = 3
        dim_x = int(nx+2*M)
        freq = np.array([1+0j,2+0j], dtype=np.complex64)

        op_forw = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)
        op_back = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

        #only the frequencies of spatial point 1 for depth 0 are non-zeros
        for l in range(nextrap):
            for i in range(nw):
                op_forw[l,i,3,:] = np.array([freq[i], freq[i], freq[i]])
                op_back[l,i,3,:] = np.array([-freq[i], -freq[i], -freq[i]])

        pulse_forw = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)
        pulse_back = np.array([[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], dtype=np.complex64)

        image = extrapolate(nextrap, nz, nx, nw, M, \
        op_forw, op_back, pulse_forw, pulse_back)

        self.assertAlmostEqual(image[0,3], -1898, delta=sdelta)
        self.assertAlmostEqual(image[1,3], 7298, delta=sdelta)

        #check that all other values are zero
        for i in range(nx):
            if i != 3:
                self.assertAlmostEqual(0.0, image[0,i], delta=sdelta)
                self.assertAlmostEqual(0.0, image[1,i], delta=sdelta)
