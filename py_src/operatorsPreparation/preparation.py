import numpy as np
import cmath
from numpy.linalg import pinv
import time
import numba
import util

#   ------------------------------------------

@numba.njit
def makeForwPSoperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = -1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( -1j*kz*dz )
            j += 1
        i += 1
    
    return psOp

#   ------------------------------------------

@numba.njit
def makeBackPSoperators(kappa, wavenumbers, dz):
    psOp = np.zeros( (len(kappa), len(wavenumbers)), dtype=np.complex64 )
    i=0
    for k in kappa:
        j=0
        for kx in wavenumbers:
            if np.abs(k) >= np.abs(kx):
                kz = np.sqrt( k**2 - kx**2 )
            else:
                kz = 1j*np.sqrt( kx**2 - k**2 )
            psOp[i,j] = cmath.exp( +1j*kz*dz )
            j += 1
        i += 1

    return psOp

#   ------------------------------------------

@numba.njit
def makeGammaOperators(wavenumbers, spatial_locations):
    gamma = np.zeros( (len(wavenumbers), len(spatial_locations)), dtype=np.complex64)
    i=0
    for kx in wavenumbers:
        j=0
        for x in spatial_locations:
            gamma[i,j] = cmath.exp( 1j*kx*x )
            j+=1
        i+=1

    return gamma

#   ------------------------------------------
#   CREATE THE ASYMMETRIC OPERATORS

def createWLSQoperators(N,M,kx,x):
    length_M = 2*M+1
    length_N = 2*N+1
    gamma = np.zeros((length_M, length_N,length_M), dtype=np.complex64)
    gammaH = np.zeros((length_M, length_M,length_N), dtype=np.complex64)
    op = np.zeros((length_M, length_M, length_N), dtype=np.complex64)

    #CREATE THE LEFT SIDE ASYMMETRIC OPERATORS
    for g in range(M):
        for n in range(length_N):
            for m in range(length_M):
                if m > M - g - 1:
                    gamma[g,n,m] = cmath.exp( 1j*kx[n]*x[m])
        gammaH[g] = np.transpose(np.conjugate(gamma[g]))

    #CREATE THE RIGHT SIDE ASYMMETRIC OPERATORS
    for g in range(M):
        for n in range(length_N):
            for m in range(length_M):
                if m < M+M-g:
                    gamma[M+g+1,n,m] = cmath.exp( 1j*kx[n]*x[m])
        gammaH[M+g+1] = np.transpose(np.conjugate(gamma[M+g+1]))

    #CREATE THE MIDDLE SYMMETRIC OPERATOR
    for n in range(length_N):
        for m in range(length_M):
            gamma[M,n,m] = cmath.exp( 1j*kx[n]*x[m])
    gammaH[M] = np.transpose(np.conjugate(gamma[M]))

    #USE GAMMA OPERATORS TO PREPARE WLSQ OPERATORS
    for g in range(length_M):
        op[g] = np.matmul(np.linalg.pinv(np.matmul(gammaH[g], gamma[g])), gammaH[g])

    return op

#   ------------------------------------------

def makeTableOfOperators(nextrap, nx, nw, k, kx, w, velmod, gIdx, opWLSQ, psOp):
    length_M = opWLSQ.shape[0]
    psOp_fs = np.zeros((nextrap, nw, nx, length_M), dtype=np.complex64)

    t0 = time.time()
    for l in range(nextrap):
        for i in range(nx):
            for j in range(nw):
                Idx = util.find_nearest(k, w[j]/velmod[l,i])
                psOp_fs[l,j,i,:] = np.matmul(opWLSQ[gIdx[i],:,:] , psOp[Idx,:])
        if __debug__:
            if l % 10 == 0:
                print("l =",l,":",round(time.time()-t0,2),"seconds.")

    return psOp_fs



