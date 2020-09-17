
#include "cuda_runtime.h"
#include <iostream>
#include "stdio.h"
#include "types.h"
#include <cmath>

extern "C"
{

__global__ void extrap_kernel(float * new_real, float * new_imag, \
    int nx, int M, \
    float * w_op_real, float * w_op_imag, \
    float * old_real, float * old_imag)
{
    int length_M = 2*M+1;
    int dim_x = nx+2*M;

    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int locIdx = length_M * Idx;

    extern __shared__ float sh[];
    float * s_old_real = sh;
    float * s_old_imag = &sh[dim_x];
    
    float ac = 0.0, bd = 0.0, ad = 0.0, bc = 0.0;
    
    if(Idx < nx){
        s_old_real[Idx] = old_real[Idx];
        s_old_imag[Idx] = old_imag[Idx];
    }
    if(Idx < 2*M){
        s_old_real[nx+Idx] = old_real[nx+Idx];
        s_old_imag[nx+Idx] = old_imag[nx+Idx];
    }

    __syncthreads();

    if(Idx < nx) {
        for(int k=0; k<length_M; ++k){
            ac += w_op_real[locIdx + k] * old_real[Idx + k];
            bd += w_op_imag[locIdx + k] * old_imag[Idx + k];
            ad += w_op_real[locIdx + k] * old_imag[Idx + k];
            bc += w_op_imag[locIdx + k] * old_real[Idx + k];
        }

        new_real[Idx] = ac - bd;
        new_imag[Idx] = ad + bc;

    }
} // end extrapolation to next depth

void extrapolate(int ns, int nextrap, int nz, int nt, int nf, int nx, int M,\
    fcomp * w_op, fcomp * pulse)
{
    //define important dimensionality parameters
    int length_M = 2*M+1;
    int dim_x = nx+2*M;
    size_t sizePulse = nf * dim_x;
    size_t sizeAllSources = ns * sizePulse;
    size_t sizeOp = nextrap * nf * nx * length_M;

    //allocate host memory
    float * h_w_op_real = new float[sizeOp];
    float * h_w_op_imag = new float[sizeOp];
    float * pulse_real = new float[sizeAllSources];
    float * pulse_imag = new float[sizeAllSources];

    //reinterpret wavefields from fcomp datatype to floats
    for(int is=0; is<ns; ++is){
        for(int j=0; j<nf; ++j){
        
            //set zeros in halo regions
            for(int i=0; i<M; ++i){
                pulse_real[is*sizePulse + j*dim_x + i] = 0.0;
                pulse_real[is*sizePulse + j*dim_x + nx + M + i] = 0.0;
                pulse_imag[is*sizePulse + j*dim_x + i] = 0.0;
                pulse_imag[is*sizePulse + j*dim_x + nx + M + i] = 0.0;
            }

            //copy wavefield in non-halo region
            for(int i=0; i<nx; ++i){
                int element = is*sizePulse + j*dim_x + M + i;
                pulse_real[element] = pulse[is*sizePulse + j*nx + i].real();
                pulse_imag[element] = pulse[is*sizePulse + j*nx + i].imag();
            }
        }
    }

    // reinterpret operator's values from fcomp datatype to floats
    for(int l=0; l<nextrap; ++l){
        int depthIdx = l*nf*nx*length_M;
        for(int j=0; j<nf; ++j){
            int freqIdx = j*nx*length_M;
            for(int i=0; i<nx; ++i){
                int xIdx = i*length_M;
                for(int ix=0; ix<length_M; ++ix){
                    int element = depthIdx + freqIdx + xIdx + ix;
                    h_w_op_real[element] = w_op[element].real();
                    h_w_op_imag[element] = w_op[element].imag();
                }
            }
        }
    }

    //allocate device memory
    float * d_w_op_real, * d_w_op_imag;
    float * d_old_real, * d_old_imag;
    float * d_new_real, * d_new_imag;
    cudaMalloc(&d_w_op_real, sizeOp * sizeof(float));
    cudaMalloc(&d_w_op_imag, sizeOp * sizeof(float));
    cudaMalloc(&d_old_real, sizeAllSources * sizeof(float));
    cudaMalloc(&d_old_imag, sizeAllSources * sizeof(float));
    cudaMalloc(&d_new_real, sizeAllSources * sizeof(float));
    cudaMalloc(&d_new_imag, sizeAllSources * sizeof(float));

    //copy operators on device
    cudaMemcpy(d_w_op_real, h_w_op_real, sizeOp*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_op_imag, h_w_op_imag, sizeOp*sizeof(float), cudaMemcpyHostToDevice);

    //copy wavefields on device
    cudaMemcpy(d_old_real, pulse_real, sizeAllSources*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_imag, pulse_imag, sizeAllSources*sizeof(float), cudaMemcpyHostToDevice);

    //define number of blocks and number of threads per block
    dim3 nThreads(64, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    size_t nBlocks_y = 1;
    size_t nBlocks_z = 1;
    dim3 nBlocks(nBlocks_x, nBlocks_y, nBlocks_z);
    std::cout << "nThreads: (" << nThreads.x << ", " << nThreads.y << ", " << nThreads.z << ")" << std::endl;
    std::cout << "nBlocks: (" << nBlocks.x << ", " << nBlocks.y << ", " << nBlocks.z << ")" << std::endl;

    cudaStream_t streams[nf];
    for(int j=0; j<nf; ++j)
        cudaStreamCreate(&streams[j]);

    for(int is=0; is<ns; ++is){

        for(int l=0; l<nextrap; ++l){

            int depthIdx = l*nf*nx*length_M;

            for(int j=0; j<nf; ++j){

                int freqIdx = j*nx*length_M;

                extrap_kernel<<<nBlocks, nThreads, 2*dim_x*sizeof(float), streams[j]>>> \
                    (&d_new_real[is*sizePulse + j*dim_x + M], &d_new_imag[is*sizePulse + j*dim_x + M], nx, M, \
                    &d_w_op_real[depthIdx+freqIdx], &d_w_op_imag[depthIdx+freqIdx], \
                    &d_old_real[is*sizePulse + j*dim_x], &d_old_imag[is*sizePulse + j*dim_x]);
            
            }
        }
    }

    for(int j=0; j<nf; ++j)
        cudaStreamDestroy(streams[j]);

    //bring wavefields back to host
    cudaMemcpy(pulse_real, d_new_real, sizeAllSources * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pulse_imag, d_new_imag, sizeAllSources * sizeof(float), cudaMemcpyDeviceToHost);

    //read real and imaginary part
    for(int is=0; is<ns; ++is){

        for(int j=0; j<nf; ++j){
        
            //copy wavefield back to fcomp
            for(int i=0; i<nx; ++i){
                int element = is*sizePulse + j*dim_x + M + i;
                pulse[is*sizePulse + j*nx + i].real(pulse_real[element]);
                pulse[is*sizePulse + j*nx + i].imag(pulse_imag[element]);
            }
        }
    }

    //free device memory
    cudaFree(d_w_op_real);
    cudaFree(d_w_op_imag);
    cudaFree(d_new_real);
    cudaFree(d_new_imag);
    cudaFree(d_old_real);
    cudaFree(d_old_imag);
}

} //end extern "C"
