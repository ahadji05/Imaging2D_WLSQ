
#include "cuda_runtime.h"
#include <iostream>
#include "stdio.h"
#include "wfPad.h"
#include <cmath>
#include <vector>
#include "revOp.h"

extern "C"
{

/*
------------------------------------------------------------------
*/
__global__ void copyPadded(fcomp * paste, fcomp * copyied, int nf, int nx, int M)
{
    int dim_x = nx+2*M;
    int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(pixelIdx_x < nx && pixelIdx_y < nf){
        int pixelIdx = pixelIdx_y * dim_x + pixelIdx_x + M;
        paste[pixelIdx] = copyied[pixelIdx];
    }
}

/*
------------------------------------------------------------------
*/
__global__ void imaging(fcomp * image, fcomp * forw_pulse, fcomp * back_pulse, int nf, int nx, int M)
{
    int dim_x = nx+2*M;
    int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

    fcomp conv;

    for(int j=0; j<nf; j++){
        int Idx = j * dim_x + pixelIdx_x + M;
        conv += forw_pulse[Idx] * thrust::conj(back_pulse[Idx]);
    }

    image[pixelIdx_x] = conv;
}

/*
------------------------------------------------------------------
*/
__global__ void extrapDepth(fcomp * new_wf, int nf, int nx, \
    int M, fcomp * w_op, fcomp * old_wf)
{
    int dim_x = nx+2*M;
    int length_M = 2*M+1;

    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    fcomp pixel = fcomp(0.0,0.0);

    if(xIdx < nx && fIdx < nf){

        for(int k=0; k<length_M; ++k){
            pixel += w_op[fIdx*nx*length_M + k*nx + xIdx] * \
                old_wf[fIdx*dim_x + xIdx + k];
        }

        new_wf[fIdx*dim_x + M + xIdx] = pixel;
    }

} // end extrapolation to next depth


/*
------------------------------------------------------------------
*/
void extrapolate(int ns, int nextrap, int nz, int nt, int nf, int nx, int M,\
    fcomp * w_op, fcomp * pulse)
{
    //define important dimensionality parameters
    int length_M = 2*M+1;
    int dim_x = nx+2*M;
    size_t sizePulse = nf * dim_x;
    size_t sizeAllSources = ns * sizePulse;
    size_t sizeOp = nextrap * nf * nx * length_M;

    //allocate and read wavefields
    std::vector<wfpad> h_pulses(ns);
    for(int is=0; is<ns; ++is)
        h_pulses[is] = wfpad(nf, nx, 1, M, 0, &pulse[is*nt*nx]);
    
    //rearrange operators
    fcomp * h_w_op = reverseOperator(w_op, nextrap, nf, nx, length_M); //reverse operator's last two indices on host

    //allocate device memory
    fcomp * d_w_op, * d_old, * d_new;
    cudaMalloc(&d_w_op, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_old, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_new, sizeAllSources * sizeof(fcomp));
    
    //copy operators on device
    cudaMemcpy(d_w_op, h_w_op, sizeOp*sizeof(fcomp), cudaMemcpyHostToDevice);

    //define number of blocks and number of threads per block
    //define number of blocks and number of threads per block
    dim3 nThreads(64, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    size_t nBlocks_y = nf;
    size_t nBlocks_z = 1;
    dim3 nBlocks(nBlocks_x, nBlocks_y, nBlocks_z);
    std::cout << "nThreads: (" << nThreads.x << ", " << nThreads.y << ", " << nThreads.z << ")" << std::endl;
    std::cout << "nBlocks: (" << nBlocks.x << ", " << nBlocks.y << ", " << nBlocks.z << ")" << std::endl;

    //create one stream per source
    cudaStream_t streams[ns];

    for(int is=0; is<ns; ++is){

        cudaStreamCreate(&streams[is]);

        cudaMemcpyAsync(&d_old[is*sizePulse], h_pulses[is].wf, sizePulse*sizeof(fcomp), \
            cudaMemcpyHostToDevice, streams[is]);
    
        for(int l=0; l<nextrap; ++l){

            int depthIdx = l*nf*nx*length_M;

            extrapDepth<<<nBlocks, nThreads>>>(&d_new[is*sizePulse], nf, nx, \
                M, &d_w_op[depthIdx], &d_old[is*sizePulse]);

        }

        cudaMemcpyAsync(h_pulses[is].wf, &d_new[is*sizePulse], sizePulse*sizeof(fcomp), \
            cudaMemcpyDeviceToHost, streams[is]);

        cudaStreamDestroy(streams[is]);
    }

    //copy to unpadded memory
    for(int is=0; is<ns; ++is)
        for (int j=0; j<nf; ++j)
            for (int i=0; i<nx; ++i)
                pulse[is*sizePulse + j*nx + i] = h_pulses[is].wf[j*dim_x + i + M];

    //free host memory
    delete [] h_w_op;

    //free device memory
    cudaFree(d_w_op);
    cudaFree(d_new);
    cudaFree(d_old);
}

} //end extern "C"
