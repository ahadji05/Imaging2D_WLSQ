
import pandas as pd
from math import ceil, pi
import numpy as np

class problemConfig:

    #constructor
    def __init__(self, filename, nz, nx, ny):
        self.filename = filename
        self.nz = int(nz)
        self.nx = int(nx)
        self.ny = int(ny)

        df = pd.read_csv(self.filename, delimiter="_")
        df = df.set_index("key") #index input by "key"

        self.nt = int(df.loc["nt"][0])
        self.nw = int(df.loc["nw"][0])
        if self.nt < self.nw:
            print("error in problemConfig: nt must by greater or equal to nw!")
            exit

        self.zmin = df.loc["zmin"][0]
        self.zmax = df.loc["zmax"][0]
        self.dz = (self.zmax-self.zmin)/self.nz

        self.zinit = df.loc["zinit"][0]
        self.zfinal = df.loc["zfinal"][0]
        self.nextrap = ceil( (self.zfinal - self.zinit)/self.dz )

        self.xmin = df.loc["xmin"][0]
        self.xmax = df.loc["xmax"][0]
        self.dx = (self.xmax-self.xmin)/self.nx

        self.tmin = df.loc["tmin"][0]
        self.tmax = df.loc["tmax"][0]
        self.dt = (self.tmax-self.tmin) / self.nt

        self.M = int(df.loc["M"][0])
        self.length_M = 2*self.M+1
        self.N = int((self.nx-1)/2)
        self.length_N = 2*self.N+1

        self.dkx = 2*pi*1/float( self.xmax - self.xmin )
        self.kx = np.fft.fftfreq(self.nx)*self.nx*self.dkx

        self.dw = 2*pi*1/float( self.tmax - self.tmin )
        self.w = np.fft.fftfreq(self.nt)*self.nt*self.dw
        self.wmax = +(0.5*self.nt-1)*self.dw

        self.xminlocal = -self.M*self.dx
        self.xmaxlocal = +self.M*self.dx
        self.xilocal = np.array([(self.xminlocal)+i*self.dx for i in range(self.length_M)], dtype=np.float32)

    #display problem configuration on the screen
    def dispInfo(self):
        print("Problem configuration:")
        print("z-range: [",round(self.zmin,0),",",round(self.zmax,0),"] meters.")
        print("     -nz:",self.nz,", dz =",round(self.dz,2),"meters")
        print("x-range: [",round(self.xmin,0),",",round(self.xmax,0),"] meters.")
        print("     -nx:",self.nx,", dx =",round(self.dx,2),"meters")
        print("local x-range: [",round(self.xminlocal,0),",",round(self.xmaxlocal,0),"]")
        print("     -nxlocal:",2*self.M + 1)
        print("     -M:",self.M)
        print("     -N:",self.N)
        print("time steps:", self.nt,", dt =",round(1000*self.dt,2),"ms")
        print("number of extrapolations:", self.nextrap)
        print("number of propagation frequencies:",self.nw)
