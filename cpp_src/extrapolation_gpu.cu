
#include "cuda_runtime.h"
#include <thrust/complex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "wfPad.h"

extern "C"
{

typedef thrust::complex<float> fcomp;

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

__global__ void extrapDepths(fcomp * forw_pulse_new, fcomp * back_pulse_new, \
    int nf, int nx, int M, \
    fcomp * forw_w_op, fcomp * back_w_op, \
    fcomp * forw_pulse_old, fcomp * back_pulse_old)
{
    int length_M = 2*M+1;
    int dim_x = nx+2*M;
    int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx_y = blockIdx.y * blockDim.y + threadIdx.y;

    fcomp pixel_forw = fcomp(0.0,0.0); //set pixels into register memory for faster access
    fcomp pixel_back = fcomp(0.0,0.0);

    if(pixelIdx_x < nx && pixelIdx_y < nf){

        int op_loc = pixelIdx_x * length_M;// operator's spatial location index
        int op_freq = pixelIdx_y * nx * length_M;// operator's frequency index
        int opIdx = op_loc + op_freq; //operator's starting location

        for (int k=0; k<length_M; ++k){
            int elemIdx = pixelIdx_y * dim_x + pixelIdx_x + k;
            pixel_forw += forw_w_op[opIdx + k] * forw_pulse_old[elemIdx];
            pixel_back += back_w_op[opIdx + k] * back_pulse_old[elemIdx];
        }

        int pixelIdx = pixelIdx_y * dim_x + pixelIdx_x + M;
        forw_pulse_new[pixelIdx] = pixel_forw;
        back_pulse_new[pixelIdx] = pixel_back;
    
    }

} // end extrapolation to next depth

void extrapolate(int ns, int nextrap, int nz, int nt, int nf, int nx, int M,\
    fcomp * w_op_forw, fcomp * forw_pulse, fcomp * w_op_back, fcomp * back_pulse,\
    float * image)
{
    //define important dimensionality parameters
    int length_M = 2*M+1;
    int dim_x = nx+2*M;
    size_t sizePulse = nf * dim_x;
    size_t sizeAllSources = ns * sizePulse;
    size_t sizeImage = nz * nx;
    size_t sizeAllImages = ns * sizeImage;
    size_t sizeOp = nextrap * nf * nx * length_M;

    //allocate host memory
    fcomp * h_image = new fcomp[sizeAllImages];
    std::vector<wfpad> h_forw_pulses(ns);
    std::vector<wfpad> h_back_pulses(ns);
    for(int is=0; is<ns; ++is){
        h_forw_pulses[is] = wfpad(nf, nx, 1, M, 0, &forw_pulse[is*nt*nx]);
        h_back_pulses[is] = wfpad(nf, nx, 1, M, 0, &back_pulse[is*nt*nx]);
    }

    //define device pointers and allocate memory
    fcomp * d_forw_pulse, * d_forw_pulse_new;
    fcomp * d_back_pulse, * d_back_pulse_new;
    fcomp * d_w_op_forw, * d_w_op_back;
    fcomp * d_image;
    cudaMalloc(&d_forw_pulse, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_forw_pulse_new, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_back_pulse, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_back_pulse_new, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_w_op_forw, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_w_op_back, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_image, sizeAllImages * sizeof(fcomp));

    //copy operators and wavefields on device
    cudaMemcpy(d_w_op_forw, w_op_forw, sizeOp * sizeof(fcomp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_op_back, w_op_back, sizeOp * sizeof(fcomp), cudaMemcpyHostToDevice);

    //define number of blocks and number of threads per block
    dim3 nThreads(16, 1, 1);

    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    size_t nBlocks_y = nf % nThreads.y == 0 ? size_t(nf/nThreads.y) : size_t(1 + nf/nThreads.y);
    size_t nBlocks_z = 1;
    dim3 nBlocks(nBlocks_x, nBlocks_y, nBlocks_z);

    cudaStream_t streams[ns];

    std::cout << "nThreads: (" << nThreads.x << ", " << nThreads.y << ", " << nThreads.z << ")" << std::endl;
    std::cout << "nBlocks: (" << nBlocks.x << ", " << nBlocks.y << ", " << nBlocks.z << ")" << std::endl;

    for(int is=0; is<ns; ++is){

        cudaStreamCreate(&streams[is]);

        cudaMemcpyAsync(&d_forw_pulse[is*sizePulse], h_forw_pulses[is].wf, \
            sizePulse*sizeof(fcomp), cudaMemcpyHostToDevice, streams[is]);
        
        cudaMemcpyAsync(&d_back_pulse[is*sizePulse], h_back_pulses[is].wf, \
            sizePulse*sizeof(fcomp), cudaMemcpyHostToDevice, streams[is]);
        
        for(int l=0; l<nextrap; ++l){

            int depthIdx = l*nx*nf*length_M;

            extrapDepths<<<nBlocks, nThreads, 0, streams[is]>>>(&d_forw_pulse_new[is*sizePulse], &d_back_pulse_new[is*sizePulse],\
                nf, nx, M, &d_w_op_forw[depthIdx], &d_w_op_back[depthIdx], &d_forw_pulse[is*sizePulse], &d_back_pulse[is*sizePulse]);
            
            imaging<<<1, nx, 0, streams[is]>>>(&d_image[is*sizeImage + l*nx], \
                &d_forw_pulse_new[is*sizePulse], &d_back_pulse_new[is*sizePulse],\
                nf, nx, M);
            
            copyPadded<<<nBlocks, nThreads, 0, streams[is]>>>(&d_forw_pulse[is*sizePulse], &d_forw_pulse_new[is*sizePulse],\
                nf, nx, M);

            copyPadded<<<nBlocks, nThreads, 0, streams[is]>>>(&d_back_pulse[is*sizePulse], &d_back_pulse_new[is*sizePulse],\
                nf, nx, M);
        }

        cudaMemcpyAsync(&h_image[is*sizeImage], &d_image[is*sizeImage], \
            sizeImage*sizeof(fcomp), cudaMemcpyDeviceToHost, streams[is]);

        cudaStreamDestroy(streams[is]);
    }

    //take real part of images
    for(int is=0; is<ns; ++is){
        for(int l=0; l<nextrap; ++l){
            for(int i=0; i<nx; ++i){
                image[is*sizeImage + l*nx + i] = reinterpret_cast<float*>(h_image)[2*(is*sizeImage + l*nx + i)];
            }
        }
    }

    //free device memory
    cudaFree(d_forw_pulse);
    cudaFree(d_forw_pulse_new);
    cudaFree(d_back_pulse);
    cudaFree(d_back_pulse_new);
    cudaFree(d_w_op_forw);
    cudaFree(d_w_op_back);
    cudaFree(d_image);

    delete [] h_image;

} // end extrapPaddedZerosAndImaging

} //end extern "C"
