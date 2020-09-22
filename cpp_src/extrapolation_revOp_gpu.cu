
#include "cuda_runtime.h"
#include <iostream>
#include "stdio.h"
#include "wfPad.h"
#include <cmath>
#include <vector>
#include "revOp.h"

#include "timer.h"
#include <chrono>
#include <ctime>

extern "C"
{

/*
-----------------------------------------------------------------
*/
__global__ void copyPadded(fcomp * paste, fcomp * copyied, \
    int nf, int nx, int M)
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
__global__ void imaging(fcomp * image, fcomp * forw_pulse, fcomp * back_pulse, \
    int nf, int nx, int M)
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
    fcomp * w_op_forw, fcomp * forw_pulse, fcomp * w_op_back, fcomp * back_pulse,\
    float * image)
{
    //define important dimensionality parameters
    int length_M = 2*M+1;
    int dim_x = nx+2*M;

    size_t sizePulse = nf * dim_x;
    size_t sizeAllSources = ns * sizePulse;
    size_t sizeOp = nextrap * nf * nx * length_M;
    size_t sizeImage = nz * nx;
    size_t sizeAllImages = ns * sizeImage;

    //rearrange operators
    timer t0("REARRANGE OPERATORS");
    fcomp * h_w_op_forw = reverseOperator(w_op_forw, nextrap, nf, nx, length_M, t0); //reverse operator's last two indices on host
    fcomp * h_w_op_back = reverseOperator(w_op_back, nextrap, nf, nx, length_M, t0); //reverse operator's last two indices on host

    //allocate device memory
    fcomp * d_image;
    cudaMalloc(&d_image, sizeAllImages * sizeof(fcomp));

    fcomp * d_w_op_forw, * d_old_forw, * d_new_forw;
    cudaMalloc(&d_w_op_forw, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_old_forw, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_new_forw, sizeAllSources * sizeof(fcomp));

    fcomp * d_w_op_back, * d_old_back, * d_new_back;
    cudaMalloc(&d_w_op_back, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_old_back, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_new_back, sizeAllSources * sizeof(fcomp));
    
    //copy operators on device
    cudaMemcpy(d_w_op_forw, h_w_op_forw, sizeOp*sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_op_back, h_w_op_back, sizeOp*sizeof(fcomp), cudaMemcpyHostToDevice);

    timer t1("CONSTRUCT PADDED WAVEFIELDS");
    t1.start();
    //allocate and read wavefields
    fcomp * h_image = new fcomp[sizeAllImages];
    std::vector<wfpad> h_forw_pulses(ns);
    std::vector<wfpad> h_back_pulses(ns);
    for(int is=0; is<ns; ++is){
        h_forw_pulses[is] = wfpad(nf, nx, 1, M, 0, &forw_pulse[is*nt*nx]);
        h_back_pulses[is] = wfpad(nf, nx, 1, M, 0, &back_pulse[is*nt*nx]);
    }
    t1.stop();

    //define number of blocks and number of threads per block
    //define number of blocks and number of threads per block
    dim3 nThreads(32, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    size_t nBlocks_y = nf % nThreads.y == 0 ? size_t(nf/nThreads.y) : size_t(1 + nf/nThreads.y);
    size_t nBlocks_z = 1;
    dim3 nBlocks(nBlocks_x, nBlocks_y, nBlocks_z);
    std::cout << "nThreads: (" << nThreads.x << ", " << nThreads.y << ", " << nThreads.z << ")" << std::endl;
    std::cout << "nBlocks: (" << nBlocks.x << ", " << nBlocks.y << ", " << nBlocks.z << ")" << std::endl;

    //create one stream per source
    cudaStream_t streams[ns];

    timer t5("EXTRAPOLATION AND IMAGING");
    t5.start();
    for(int is=0; is<ns; ++is){

        cudaStreamCreate(&streams[is]);

        cudaMemcpyAsync(&d_old_forw[is*sizePulse], h_forw_pulses[is].wf, \
            sizePulse*sizeof(fcomp), cudaMemcpyHostToDevice, streams[is]);
        cudaMemcpyAsync(&d_old_back[is*sizePulse], h_back_pulses[is].wf, \
            sizePulse*sizeof(fcomp), cudaMemcpyHostToDevice, streams[is]);

        for(int l=0; l<nextrap; ++l){

            int depthIdx = l*nf*nx*length_M;

            extrapDepth<<<nBlocks, nThreads, 0, streams[is]>>>(&d_new_forw[is*sizePulse], nf, nx, \
                M, &d_w_op_forw[depthIdx], &d_old_forw[is*sizePulse]);
            
            extrapDepth<<<nBlocks, nThreads, 0, streams[is]>>>(&d_new_back[is*sizePulse], nf, nx, \
                M, &d_w_op_back[depthIdx], &d_old_back[is*sizePulse]);
            
            imaging<<<1, nx>>>(&d_image[is*sizeImage + l*nx], &d_new_forw[is*sizePulse], \
                &d_new_back[is*sizePulse], nf, nx, M);
            
            copyPadded<<<nBlocks, nThreads>>>(&d_old_forw[is*sizePulse], &d_new_forw[is*sizePulse],\
                nf, nx, M);

            copyPadded<<<nBlocks, nThreads>>>(&d_old_back[is*sizePulse], &d_new_back[is*sizePulse],\
                nf, nx, M);
            
        }
        cudaMemcpyAsync(h_forw_pulses[is].wf, &d_new_forw[is*sizePulse], \
            sizePulse*sizeof(fcomp), cudaMemcpyDeviceToHost, streams[is]);
        cudaMemcpyAsync(h_back_pulses[is].wf, &d_new_back[is*sizePulse], \
            sizePulse*sizeof(fcomp), cudaMemcpyDeviceToHost, streams[is]);
        cudaMemcpyAsync(&h_image[is*sizeImage], &d_image[is*sizeImage], \
            sizeImage*sizeof(fcomp), cudaMemcpyDeviceToHost, streams[is]);

        cudaStreamDestroy(streams[is]);
    }
    cudaDeviceSynchronize();
    t5.stop();

    timer t2("WRITE-BACK UNPADDED WAVEFIELDS");
    t2.start();
    //copy to unpadded memory
    for(int is=0; is<ns; ++is)
        for (int j=0; j<nf; ++j)
            for (int i=0; i<nx; ++i){
                forw_pulse[is*sizePulse + j*nx + i] = h_forw_pulses[is].wf[j*dim_x + i + M];
                back_pulse[is*sizePulse + j*nx + i] = h_back_pulses[is].wf[j*dim_x + i + M];
            }
    t2.stop();

    timer t3("READ IMAGES");
    t3.start();
    //take real part of images
    for(int is=0; is<ns; ++is)
        for(int l=0; l<nextrap; ++l)
            for(int i=0; i<nx; ++i){
                image[is*sizeImage + l*nx + i] = reinterpret_cast<float*>(h_image)[2*(is*sizeImage + l*nx + i)];
            }
    t3.stop();

    timer t4("FREE HOST MEMORY");
    t4.start();
    //free host memory
    delete [] h_image;
    delete [] h_w_op_forw;
    delete [] h_w_op_back;
    t4.stop();

    //free device memory
    cudaFree(d_w_op_forw);
    cudaFree(d_w_op_back);
    cudaFree(d_new_forw);
    cudaFree(d_old_forw);
    cudaFree(d_new_back);
    cudaFree(d_old_back);
    cudaFree(d_image);

    std::cout << "---Timer info---" << std::endl;
    std::cout << "----------------" << std::endl;
    t0.dispInfo();
    t1.dispInfo();
    t2.dispInfo();
    t3.dispInfo();
    t4.dispInfo();
    t5.dispInfo();

}

} //end extern "C"
