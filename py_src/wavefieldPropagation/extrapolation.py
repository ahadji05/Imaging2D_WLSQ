
from numba import njit
import numpy as np

#--------------------------------------------------
# copy data from one numpy.complex64 2D array to another.
# This operation is manually implemented because np.copyto()
# was not supported by numba when this code was written.
@njit
def c64copy2d(arr, nrows, ncols):
    cparr = np.zeros((nrows, ncols), dtype=arr.dtype)
    for i in range(nrows):
        for j in range(ncols):
            cparr[i,j] = arr[i,j]
    
    return cparr

#----------------------------------------------------------
# element by element multiplication of 2D numpy arrays
# and then reduction across columns! This operation is equivalent
# to np.add.reduce(arr1 * arr2, axis=0).
@njit
def imaging(arr1, arr2, nrows, ncols):
    conv = np.zeros((nrows, ncols), dtype=arr1.dtype)

    for i in range(nrows):
        for j in range(ncols):
            conv[i,j] = arr1[i,j]*arr2[i,j]
    
    image = np.zeros(ncols, dtype=arr1.dtype)
    for i in range(ncols):
        image[i] = np.sum(conv[:,i])

    return image.real

#--------------------------------------------------------------------------------------
# propgation of two wavefields using padded zeros in halo regions
@njit
def extrapolate(nextrap, nz, nx, nf, M, w_op_fs_forw, w_op_fs_back, wave_fs, shot_fs):

    if nz < nextrap:
        print("nz must be equal or greater than nextrap!")

    nt = wave_fs.shape[0]
    length_M = int(2*M+1)
    dim_x = int(nx+2*M)

    old_forw = np.zeros((nt,dim_x), dtype=wave_fs.dtype)
    old_forw[:,M:nx+M] = c64copy2d(wave_fs, nt, nx)
    old_back = np.zeros((nt,dim_x), dtype=shot_fs.dtype)
    old_back[:,M:nx+M] = c64copy2d(shot_fs, nt, nx)

    new_forw = np.zeros_like(old_forw)
    new_back = np.zeros_like(old_back)
    image = np.zeros((nz,nx), dtype=np.float32)

    for l in range(nextrap):
        for j in range(nf):
            for i in range(nx):
                new_forw[j,i+M] = np.dot(w_op_fs_forw[l,j,i,:], old_forw[j,i:i+length_M])
                new_back[j,i+M] = np.dot(w_op_fs_back[l,j,i,:], old_back[j,i:i+length_M])
        old_forw = c64copy2d(new_forw, nt, dim_x)
        old_back = c64copy2d(new_back, nt, dim_x)
        image[l,:] = imaging(new_forw[:,M:nx+M], np.conj(new_back[:,M:nx+M]), nt, nx)

    return image
