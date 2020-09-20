
import sys

import numpy as np
import scipy.fftpack as spfft
from scipy import ndimage

import readShotFiles

sys.path.append('py_src/operatorsPreparation/')
import preparation as prep
import problemConfig as pConfig
import util

sys.path.append('./cpp_src/')
from interface_cuda import extrapolate as extrap_device
from interface_cuda import extrapolate_revOp as extrap_device_revOp
from interface import extrapolate as extrap_host

import time

time_start = time.time()

#   ------------------------------------------
#   READ COMMAND LINE ARGUMENTS
io_time_start = time.time()

file_vel = sys.argv[1] #filename of velocity model (csv)
file_config = sys.argv[2] #filename of configuration
shots_dir = sys.argv[3] #directory to shots
path_to_output = sys.argv[4] #path to output directory
option = sys.argv[5] #select either host or device

#   ------------------------------------------
#   READ VELOCITY MODEL

velocity_model = np.genfromtxt(file_vel, delimiter=',')
slownes = 1.0 / velocity_model
slownes = ndimage.gaussian_filter(slownes, sigma=2)
velocity_model = 1.0 / slownes
np.savetxt(path_to_output+"/smoothed_velmod.csv", velocity_model, delimiter=',')

#   ------------------------------------------
#   SETUP EXPERIMENT CONFIGURATION

config = pConfig.problemConfig(filename=file_config, \
    nz=velocity_model.shape[0],\
    nx=velocity_model.shape[1], ny=1)

io_time_stop = time.time()
io_time_total = round(io_time_stop-io_time_start,2)

#   ------------------------------------------
#   PRINT PROBLEM CONFIGURATION

config.dispInfo()

#   ------------------------------------------
#   PREPARE WLSQ OPERATORS
print("preparing wlsq operators ...")
prep_ps_time_start = time.time()

op = prep.createWLSQoperators(config.N, config.M, config.kx, config.xilocal)
gIndices = util.defWLSQIndices(nx=config.nx, M=config.M, flip=False, sym=True, extent=5)

#   ------------------------------------------
#   PREPARE PHASE-SHIFT OPERATORS

maxvel = np.amax(velocity_model)
minvel = np.amin(velocity_model)
kappa = util.createKappa(maxvel, minvel, config.dw, config.wmax, 0.25)
print("    using",len(kappa),"Phase-Shift operators.")

w_op_fk_forw = prep.makeForwPSoperators(kappa, config.kx, config.dz)

w_op_fk_back = prep.makeBackPSoperators(kappa, config.kx, config.dz)

prep_ps_time_stop = time.time()
prep_ps_time_total = round(prep_ps_time_stop-prep_ps_time_start,2)

#   ------------------------------------------
#   PREPARE TABLES OF OPERATORS
print("preparing tables of operators ...")
prep_tables_time_start = time.time()

w_op_fs_forw = prep.makeTableOfOperators(config.nextrap, config.nx,\
    config.nw, kappa, config.kx, config.w,\
    velocity_model, gIndices, op, w_op_fk_forw)

w_op_fs_back = prep.makeTableOfOperators(config.nextrap, config.nx,\
    config.nw, kappa, config.kx, config.w,\
    velocity_model, gIndices, op, w_op_fk_back)

prep_tables_time_stop = time.time()
prep_tables_time_total = round(prep_tables_time_stop-prep_tables_time_start,2)

#   ------------------------------------------
#   READ SEISMOGRAPH FILES AND SOURCES INDICES
print("preparing shots for all sources ...")

prep_shots_time_start = time.time()

shot_isx, file_shot = readShotFiles.returnShotIndices(shots_dir, "csv", "seis", "_")
ns = len(shot_isx)

#   ------------------------------------------
#   PREPARE RICKER WAVELETS PER SHOT

v = velocity_model[0,0] #assumes const velocity in first depth-step
tj = [j*config.dt for j in range(config.nt)]
xi = [config.xmin+(i+1)*config.dx for i in range(config.nx)]

pulse_forw_st = np.zeros((ns,config.nt,config.nx), dtype=np.float32)
pulse_forw_fs = np.zeros((ns,config.nt,config.nx), dtype=np.complex64)
pulse_back_st = np.zeros((ns,config.nt,config.nx), dtype=np.float32)
pulse_back_fs = np.zeros((ns,config.nt,config.nx), dtype=np.complex64)
for s in range(ns):
    pulse_forw_st[s,:,:] = util.makeRickerWavelet([config.xmin + shot_isx[s]*config.dx], config.zinit, config.nt, config.nx, tj, xi, config.dz, v)
    pulse_forw_fs[s,:,:] = spfft.fft(pulse_forw_st[s,:,:], axis=0)
    pulse_back_st[s,:,:] = np.genfromtxt(file_shot[s], delimiter=',', dtype=np.float32)
    pulse_back_fs[s,:,:] = spfft.fft(pulse_back_st[s,:,:], axis=0)

prep_shots_time_stop = time.time()
prep_shots_time_total = round(prep_shots_time_stop-prep_shots_time_start,2)

#   -------------------------------------------
#   EXTRAPOLATION AND IMAGING
print("extrapolation and imaging ...")
print("ns:",ns)
extrap_time_start = time.time()

image = np.zeros((ns,config.nz,config.nx), dtype=np.float32)
#---------------------------------------------------------------------
if option == "device":
    print("Extrapolation on device")
    extrap_device(ns, config.nextrap, config.nz, config.nt, config.nw, \
        config.nx, config.M, w_op_fs_forw, pulse_forw_fs, w_op_fs_back, \
        pulse_back_fs, image)
elif option == "device_revOp":
    print("Extrapolation on device (rearranged operators)")
    extrap_device_revOp(ns, config.nextrap, config.nz, config.nt, config.nw, \
        config.nx, config.M, w_op_fs_forw, pulse_forw_fs, w_op_fs_back, \
        pulse_back_fs, image)
elif option == "host":
    print("Extrapolation on host")
    extrap_host(ns, config.nextrap, config.nz, config.nx, config.nw, config.nt, config.M, \
        w_op_fs_forw, w_op_fs_back, pulse_forw_fs, pulse_back_fs, \
        image)
else:
    print("No extrapolation option selected!")
    print("check command line parameter 5")
#---------------------------------------------------------------------
extrap_time_stop = time.time()
extrap_time_total = round(extrap_time_stop-extrap_time_start,2)

time_stop = time.time()
total_time = round(time_stop-time_start,2)
#   ---------------------------------

print("-------------------------------")
print("Total program time (s) :",total_time)
print("    I/O-time (s)                   :",io_time_total)
print("    Prep phase-shift operators (s) :",prep_ps_time_total)
print("    Prep tables of operators (s)   :",prep_tables_time_total)
print("    Prep shots (s)                 :",prep_shots_time_total)
print("    Extrapolation and Imaging (s)  :",extrap_time_total)

#   ---------------------------------
#   OUTPUT

# final image
final_image = np.zeros((config.nz,config.nx), dtype=np.float32)
for s in range(ns):
    final_image += image[s]
    # np.savetxt(path_to_output+"/image"+str(shot_isx[s])+"_.csv", image[s], delimiter=',')

#save final image
np.savetxt(path_to_output+"/final_image.csv", final_image, delimiter=',')
