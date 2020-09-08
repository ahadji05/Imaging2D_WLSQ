
import sys
sys.path.append('../py_src/operatorsPreparation')

import unittest

import numpy as np
from math import pi

#   ------------------------------------------

from util import makeRickerWavelet as makeShot

#need to run an un-tested example in advance to avoid run-time error by numba!
shot = makeShot([0], 0.0, 4, 3, [1,2,3,4], [1,2,3], 0.1, 500)

class Test_makeRickerWavelet(unittest.TestCase):

    def test_makeRickerWavelet(self):
        
        #test-setup
        isx = -600
        nt = 500
        dt = 0.001
        nx = 151
        dx = 15
        xmin = -800 # meters
        z1 = 0; z2 = 10 #meters
        v = 2600 # meters/s

        #prep hard-coded test case
        testshot = np.zeros((nt,nx), dtype=np.float32)
        for i in range(nx):
            xloc = xmin + i*dx
            r = np.sqrt( (z2-z1)**2 + (isx-xloc)**2 )
            t0 = r/v
            for j in range(nt):        
                t_t0 = j*dt - t0
                term = pi**2*30.0**2*(t_t0)**2
                testshot[j,i] = (1-2*term)*np.exp(-term)/np.sqrt(r)

        #run function
        tj = [j*dt for j in range(nt)]
        xi = [xmin + i*dx for i in range(nx)]
        shot = makeShot([isx], z1, nt, nx, tj, xi, z2, v)

        #test results
        self.assertEqual(shot.shape[0], nt)
        self.assertEqual(shot.shape[1], nx)
        for i in range(nx):
            for j in range(nt):
                self.assertAlmostEqual(shot[j,i], testshot[j,i], delta=0.00001)

#   ------------------------------------------

from util import defWLSQIndices as indices

class Test_wlsq_indices(unittest.TestCase):

    def test_wlsq_indices_sym1(self):

        # test all symmetric (1)
        idxs = indices(nx=10, M=4, flip=False, sym=True, extent=1)
        for i in range(10):
            self.assertEqual(idxs[i], 4)
    
    def test_wlsq_indices_sym2(self):

        # test all symmetric (2)
        idxs = indices(nx=131, M=14, flip=True, sym=True, extent=5)
        for i in range(15):
            self.assertEqual(idxs[i], 14)
    
    def test_wlsq_indices_flip(self):

        # test flip asymmetric
        idxsflipped = indices(nx=11, M=5, flip=True, sym=False, extent=5)
        idxsUnflipped = indices(nx=11, M=5, flip=False, sym=False, extent=5)
        for i in range(11):
            self.assertEqual(np.flip(idxsUnflipped)[i], idxsflipped[i])

    def test_wlsq_indices_asym1(self):

        # test by value asymmetric (1)
        idxs = indices(nx=15, M=5, flip=False, sym=False, extent=4)
        self.assertEqual(idxs[0],0)
        self.assertEqual(idxs[1],1)
        self.assertEqual(idxs[2],2)
        self.assertEqual(idxs[3],3)
        for i in range(7):
            self.assertEqual(idxs[i+4], 5)
        self.assertEqual(idxs[11],7)
        self.assertEqual(idxs[12],8)
        self.assertEqual(idxs[13],9)
        self.assertEqual(idxs[14],10)

    def test_wlsq_indices_asym2(self):

        # test by value asymmetric (2)
        idxs = indices(nx=151, M=5, flip=False, sym=False, extent=5)
        self.assertEqual(idxs[0],0)
        self.assertEqual(idxs[1],1)
        self.assertEqual(idxs[2],2)
        self.assertEqual(idxs[3],3)
        self.assertEqual(idxs[4],4)
        for i in range(141):
            self.assertEqual(idxs[i+5], 5)
        self.assertEqual(idxs[147],7)
        self.assertEqual(idxs[148],8)
        self.assertEqual(idxs[149],9)
        self.assertEqual(idxs[150],10)        

#   ------------------------------------------

from util import createKappa

class Test_createKappa(unittest.TestCase):

    def test_createKappa_different_dense(self):

        minvel = 1000
        maxvel = 3000
        wmax = 650
        dw = 3
        dense = [1,2,4,0.5,0.25,0.125,0.1,0.05]

        for den in dense:
            k = createKappa( maxvel, minvel, dw, wmax, den )
            
            #check that kappa steps = dw/maxvel/den
            self.assertAlmostEqual( (k[1]-k[0]), (dw/maxvel/den), delta=0.00001 )

            #check that the first value is  zero (0)
            self.assertAlmostEqual( 0.0, k[0], delta=0.00001)

            #ensure that always the range of kappa[i] covers the max-possible k
            length = len(k)
            kmax = wmax/minvel
            self.assertGreaterEqual( k[length-1], kmax )
    
    def test_createKappa_different_minvel(self):
    
        minvel = [1000, 1500, 2000, 2500]
        maxvel = 3000
        wmax = 650
        dw = 3
        den = 0.25

        for vel in minvel:
            k = createKappa( maxvel, vel, dw, wmax, den )
            
            #check that kappa steps = dw/maxvel/den
            self.assertAlmostEqual( (k[1]-k[0]), (dw/maxvel/den), delta=0.00001 )

            #check that the first value is  zero (0)
            self.assertAlmostEqual( 0.0, k[0], delta=0.00001)

            #ensure that always the range of kappa[i] covers the max-possible k
            length = len(k)
            kmax = wmax/vel
            self.assertGreaterEqual( k[length-1], kmax )
        
    def test_createKappa_different_maxvel(self):
    
        minvel = 1000
        maxvel = [3000,4000,5000,10000]
        wmax = 836
        dw = 5.5
        den = 0.125

        for vel in maxvel:
            k = createKappa( vel, minvel, dw, wmax, den )
            
            #check that kappa steps = dw/maxvel/den
            self.assertAlmostEqual( (k[1]-k[0]), (dw/vel/den), delta=0.00001 )

            #check that the first value is  zero (0)
            self.assertAlmostEqual( 0.0, k[0], delta=0.00001)

            #ensure that always the range of kappa[i] covers the max-possible k
            length = len(k)
            kmax = wmax/vel
            self.assertGreaterEqual( k[length-1], kmax )
        
    def test_createKappa_different_wmax(self):
    
        minvel = 1000
        maxvel = 3000
        wmax = [400,500,800,1300]
        dw = 7.5
        den = 1

        for w in wmax:
            k = createKappa( maxvel, minvel, dw, w, den )
            
            #check that kappa steps = dw/maxvel/den
            self.assertAlmostEqual( (k[1]-k[0]), (dw/maxvel/den), delta=0.00001 )

            #check that the first value is  zero (0)
            self.assertAlmostEqual( 0.0, k[0], delta=0.00001)

            #ensure that always the range of kappa[i] covers the max-possible k
            length = len(k)
            kmax = w/minvel
            self.assertGreaterEqual( k[length-1], kmax )

#   ------------------------------------------

from util import find_nearest

find_nearest(np.array([1,2,3],dtype=np.int16),2)

class Test_find_nearest(unittest.TestCase):

    #test for integers
    def test_find_nearest_int16(self):

        Aint16 = np.array([0,4,6,3,2,54,3], dtype=np.int16)

        idx = find_nearest(Aint16, 6)
        self.assertEqual(idx,2)

    #test for floats
    def test_find_nearest_float32(self):
    
        Afloat32 = np.array([0.0,4.5,6.2,3.7,2.4,5.4,3000.0], dtype=np.float32)

        idx = find_nearest(Afloat32, 5.0)
        self.assertEqual(idx,5)

        idx = find_nearest(Afloat32, 4.8)
        self.assertEqual(idx,1)
