
#path to intel directory
INTEL = /opt/intel

#path to directory mkl
MKLROOT = $(INTEL)/mkl

# MKLROOT = $(EBROOTIMKL)/mkl

#define required for the linking MKL libraries
MKLLIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed\
-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#linking flags
LFLAGS= $(MKLLIB)

#inlcude Seiscmic Unix and MKL header files
INC = -DMKL_ILP64 -m64 -I$(MKLROOT)/include

#select compilers and flags
CXX=g++
CXXFLAGS = -O3 -std=c++11 $(INC)

#-----------------------------------
#	no need to edit below this line

all: mkl_fft mkl_linalg extrapolation_cpu extrapolation_gpu

mkl_fft:
	$(CXX) -shared -fPIC $(CXXFLAGS) -o mkl_fft.so mkl_fft.cpp

mkl_linalg: mkl_linalg.o
	$(CXX) -shared -fPIC -o mkl_linalg.so mkl_linalg.o $(LFLAGS)

extrapolation_cpu: extrapolation_cpu.o
	$(CXX) -shared -fPIC -o extrapolation_cpu.so extrapolation_cpu.o mkl_linalg.o $(LFLAGS) -fopenmp

extrapolation_cpu.o: mkl_linalg.o
	$(CXX) -Wall $(CXXFLAGS) -shared -fPIC -c extrapolation_cpu.cpp -fopenmp

mkl_linalg.o:
	$(CXX) -Wall $(CXXFLAGS) -shared -fPIC -c mkl_linalg.cpp

extrapolation_gpu:
	nvcc $(CXXFLAGS) -arch=sm_35 -Xcompiler -fPIC -shared -o extrapolation_gpu.so extrapolation_gpu.cu

#remove executable and object files
clean:
	rm -f *.so *.o
