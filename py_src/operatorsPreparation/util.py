import numpy as np
from numba import njit
from math import exp, pi, ceil

@njit
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

@njit
def ricker_wv_map(t_t0):
    fmax = 30 #Hz
    term = pi*pi*fmax*fmax*(t_t0)*(t_t0)
    return (1-2*term)*exp(-term)

def defWLSQIndices(nx, M, flip, sym, extent):
    gIdx = np.zeros(nx, dtype=np.int16)
    
    if sym == True:
        gIdx[:] = M
        return gIdx # all symmetric
    
    d1 = 0; d2 = M+1
    for i in range(nx):
        gIdx[i] = M # far from edges -> symmetric 
        if i < M:
            gIdx[i] = d1 # near "left" edge
            d1 += 1
        if i >= nx-M:
            gIdx[i] = d2 # near "right" edge
            d2 += 1
    gIdx[extent:nx-extent] = M

    if flip == True:
        gIdx = np.flip(gIdx) # flip sides

    return gIdx
    
def createKappa(cmax, cmin, dw, wmax, dense):
    deltak = dw/cmax/dense
    maxk = wmax/cmin
    nk = int(maxk/deltak) + ceil(cmax/cmin)
    kappa = np.array([i*deltak for i in range(nk)])

    return kappa

def makeRickerWavelet(isx, isz, nt, nx, tj, xi, init_depth, v):
    wavelet = np.zeros((nt,nx), dtype=np.float32)
    t_t0 = np.zeros(nt,dtype=np.float32)

    for sx in isx:
        for i in range(nx):
            r = np.sqrt((init_depth-isz)**2 + (xi[i]-sx)**2)
            t0 = r/v
            t_t0[:] = tj[:]-t0
            wavelet[:,i] += np.asarray(list(map(ricker_wv_map, t_t0)))/np.sqrt(r)
    
    return wavelet
